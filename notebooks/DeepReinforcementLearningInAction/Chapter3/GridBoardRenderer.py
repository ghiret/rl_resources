# File: grid_renderer.py
import contextlib
import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont  # Import ImageDraw and ImageFont

try:
    # Optional: Only needed if returning the displayable object
    from IPython.display import Image as IPImage

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# Import the GridBoard class from your existing file
try:
    from GridBoard import GridBoard
except ImportError:
    print("Error: Could not import GridBoard. Make sure GridBoard.py is accessible.")
    GridBoard = None

# --- Constants and Defaults ---
DEFAULT_COLOR_MAP = {
    "Player": (30, 144, 255),  # Dodger Blue
    "Goal": (50, 205, 50),  # Lime Green
    "Pit": (220, 20, 60),  # Crimson
    "Wall": (105, 105, 105),  # Dim Gray
    "boundary": (0, 0, 0),  # Black
}
DEFAULT_BACKGROUND_COLOR = (240, 248, 255)  # Alice Blue
DEFAULT_GRID_COLOR = (180, 180, 180)  # Light Gray
DEFAULT_GRID_WIDTH = 1

DEFAULT_PLAYER_TEXT = "P"
DEFAULT_PLAYER_TEXT_COLOR = (255, 255, 255)  # White (contrasts with blue)
DEFAULT_WALL_TEXT = "W"
DEFAULT_WALL_TEXT_COLOR = (255, 255, 255)  # White (contrasts with gray)

# Define a default font path (adjust if needed)
# Common paths: '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf' (Linux)
# Or provide a path to a font file you have downloaded
DEFAULT_FONT_PATH = (
    None  # Set to a path like "arial.ttf" or "DejaVuSans.ttf" if available
)
DEFAULT_FONT_SIZE_FACTOR = 1.0  # Font size relative to cell size


logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

MIN_BITMAP_FONT_SIZE = 10

GRIDBOARD_NOT_AVAILABLE_ERROR = "GridBoard class not available."


# --- Helper function to load font ---
def load_font(font_path=DEFAULT_FONT_PATH, size=15):
    """Load a TrueType font or fall back to default bitmap font."""
    if font_path and Path(font_path).exists():
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            print(
                f"Warning: Could not load font '{font_path}'. Falling back to default."
            )
    # Fallback to default bitmap font if path not provided, not found, or fails loading
    if size <= MIN_BITMAP_FONT_SIZE:  # Default bitmap font is small
        return ImageFont.load_default()

        # Attempt to load default again, but warn it might be small
    print(f"Warning: Using default bitmap font, which may be pixelated at size {size}.")
    return ImageFont.load_default()


# --- Rendering Functions ---


def render_color(
    board: GridBoard,
    color_map: dict = DEFAULT_COLOR_MAP,
    bg_color: tuple = DEFAULT_BACKGROUND_COLOR,
    cell_size: int = 30,
    show_grid: bool = True,
    grid_color: tuple = DEFAULT_GRID_COLOR,
    grid_width: int = DEFAULT_GRID_WIDTH,
    player_text: str = DEFAULT_PLAYER_TEXT,
    player_text_color: tuple = DEFAULT_PLAYER_TEXT_COLOR,
    wall_text: str = DEFAULT_WALL_TEXT,
    wall_text_color: tuple = DEFAULT_WALL_TEXT_COLOR,
    font_path: str = DEFAULT_FONT_PATH,
    font_size_factor: float = DEFAULT_FONT_SIZE_FACTOR,
    shape_padding_factor: float = 0.1,  # Padding for player circle (10%)
) -> Image.Image:
    """
    Render GridBoard state as color PIL Image with grid, player circle+text, wall text.

    Args:
    ----
        board: GridBoard instance.
        color_map: Dict mapping component names to RGB tuples.
        bg_color: Background RGB color.
        cell_size: Pixel size per cell.
        show_grid: Draw grid lines if True.
        grid_color: Grid line RGB color.
        grid_width: Grid line width.
        player_text: Text for Player ('P').
        player_text_color: Player text RGB color.
        wall_text: Text for Wall ('W').
        wall_text_color: Wall text RGB color.
        font_path: Path to a .ttf font file (optional).
        font_size_factor: Font size relative to cell_size (e.g., 0.6).
        shape_padding_factor: Padding around player circle (relative to cell_size).

    Returns:
    -------
        PIL Image object.

    """
    if board is None:
        raise ValueError(GRIDBOARD_NOT_AVAILABLE_ERROR)

    layers_np = board.render_np()
    num_layers, height, width = layers_np.shape

    component_names = list(board.components.keys()) + list(board.masks.keys())

    if num_layers != len(component_names) and num_layers > 0:
        logging.warning(
            f"Layer/component name mismatch ({num_layers} vs {len(component_names)})."
        )

    img_height = height * cell_size
    img_width = width * cell_size
    img_array = np.full((img_height, img_width, 3), bg_color, dtype=np.uint8)

    player_coords = []  # Store (r, c) for player
    wall_coords = []  # Store (r, c) for walls

    # Pass 1: Fill background colors for all components EXCEPT Player
    # Also store coordinates for Player and Wall
    for k in range(num_layers):
        layer = layers_np[k]
        component_name = (
            component_names[k] if k < len(component_names) else f"Unknown_{k}"
        )
        color = color_map.get(component_name, (0, 0, 0))  # Default black
        if component_name not in color_map and component_name != f"Unknown_{k}":
            logging.warning(f"No color defined for '{component_name}'. Using black.")

        rows, cols = np.nonzero(layer)
        coords = list(zip(rows, cols, strict=False))

        # Store coords for later drawing
        if component_name == "Player":
            player_coords.extend(coords)
            # *** Skip background fill for player here ***
            # We will draw the circle later
            continue
        if component_name == "Wall":
            wall_coords.extend(coords)
            # Continue to fill background for wall

        # Fill background colors for non-player components
        for r, c in coords:
            y_start, y_end = r * cell_size, (r + 1) * cell_size
            x_start, x_end = c * cell_size, (c + 1) * cell_size
            if y_end <= img_height and x_end <= img_width:
                img_array[y_start:y_end, x_start:x_end] = color

    # Convert base background array to PIL Image
    img = Image.fromarray(img_array, "RGB")
    draw = ImageDraw.Draw(img)

    # Calculate font size and load font
    font_size = max(5, int(cell_size * font_size_factor))  # Ensure minimum size 5
    font = load_font(font_path, font_size)

    # Pass 2: Draw Player Circle and Text
    if player_text and player_coords:
        player_color = color_map.get("Player", (0, 0, 0))  # Get player background color
        padding = int(cell_size * shape_padding_factor)
        for r, c in player_coords:
            # Bounding box for the circle (with padding)
            y0 = r * cell_size + padding
            x0 = c * cell_size + padding
            y1 = (r + 1) * cell_size - padding
            x1 = (c + 1) * cell_size - padding
            bbox = [(x0, y0), (x1, y1)]

            # Draw the filled circle
            draw.ellipse(
                bbox, fill=player_color, outline=None
            )  # Use player color as fill

            # Calculate center and draw text 'P'
            center_x = c * cell_size + cell_size / 2
            center_y = r * cell_size + cell_size / 2
            draw.text(
                (center_x, center_y),
                player_text,
                fill=player_text_color,
                font=font,
                anchor="mm",
            )

    # Pass 3: Draw Wall Text
    if wall_text and wall_coords:
        for r, c in wall_coords:
            center_x = c * cell_size + cell_size / 2
            center_y = r * cell_size + cell_size / 2
            draw.text(
                (center_x, center_y),
                wall_text,
                fill=wall_text_color,
                font=font,
                anchor="mm",
            )

    # Pass 4: Draw Grid Lines
    if show_grid and cell_size > 1 and grid_width > 0:
        for x in range(0, img_width + 1, cell_size):
            draw_x = min(
                x, img_width - max(1, grid_width // 2)
            )  # Adjust for width at edge
            line = [(draw_x, 0), (draw_x, img_height)]
            with contextlib.suppress(ValueError):
                draw.line(line, fill=grid_color, width=grid_width)
        for y in range(0, img_height + 1, cell_size):
            draw_y = min(
                y, img_height - max(1, grid_width // 2)
            )  # Adjust for width at edge
            line = [(0, draw_y), (img_width, draw_y)]
            with contextlib.suppress(ValueError):
                draw.line(line, fill=grid_color, width=grid_width)

    return img


def create_gif(
    board_states: list[GridBoard],
    filename: str | None = None,  # Default filename is None
    duration: int = 500,
    loop: int = 0,
    color_map: dict = DEFAULT_COLOR_MAP,
    bg_color: tuple = DEFAULT_BACKGROUND_COLOR,
    cell_size: int = 30,
    show_grid: bool = True,
    grid_color: tuple = DEFAULT_GRID_COLOR,
    grid_width: int = DEFAULT_GRID_WIDTH,
    player_text: str = DEFAULT_PLAYER_TEXT,
    player_text_color: tuple = DEFAULT_PLAYER_TEXT_COLOR,
    wall_text: str = DEFAULT_WALL_TEXT,
    wall_text_color: tuple = DEFAULT_WALL_TEXT_COLOR,
    font_path: str = DEFAULT_FONT_PATH,
    font_size_factor: float = DEFAULT_FONT_SIZE_FACTOR,
    shape_padding_factor: float = 0.1,
):
    """
    Create animated GIF from GridBoard states.

    Args:
    ----
        board_states: List of GridBoard instances to animate.
        filename: Path to save GIF. If None, returns object.
        duration: Duration for each frame in milliseconds.
        loop: Number of times to loop (0 for infinite).
        color_map: Dict mapping component names to RGB tuples.
        bg_color: Background RGB color.
        cell_size: Pixel size per cell.
        show_grid: Draw grid lines if True.
        grid_color: Grid line RGB color.
        grid_width: Grid line width.
        player_text: Text for Player ('P').
        player_text_color: Player text RGB color.
        wall_text: Text for Wall ('W').
        wall_text_color: Wall text RGB color.
        font_path: Path to a .ttf font file.
        font_size_factor: Font size relative to cell_size.
        shape_padding_factor: Padding around player circle.

    Returns:
    -------
        Optional[IPython.display.Image]: Displayable image object if filename is None
                                       and IPython is available.
        Optional[bytes]: Raw GIF bytes if filename is None and IPython is not available.
        None: If filename is provided (saves to file).

    """
    if board_states is None:
        logging.error("GridBoard class not available. Cannot create GIF.")
        return None  # Return None on error

    frames = []
    total_frames = len(board_states)
    print(f"Generating {total_frames} frames for GIF...")
    for i, board_state in enumerate(board_states):
        # Use \r to overwrite line, print space at end to clear previous longer message
        print(f"\rRendering frame {i+1}/{total_frames}... ", end="")
        img = render_color(
            board_state,
            color_map,
            bg_color,
            cell_size,
            show_grid,
            grid_color,
            grid_width,
            player_text,
            player_text_color,
            wall_text,
            wall_text_color,
            font_path,
            font_size_factor,
            shape_padding_factor,
        )
        frames.append(img)
    print("\nFrame rendering complete.")

    if not frames:
        logging.error("No frames generated, cannot create GIF.")
        return None

    # --- Save to file OR return display object ---
    if filename:
        # Save to file logic (as before)
        print(f"Saving GIF to '{filename}'...")
        try:
            frames[0].save(
                filename,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=loop,
                optimize=False,
            )
            print("GIF saved successfully.")
            return None  # Indicate success by returning None when saving file
        except Exception:
            logging.exception("Error saving GIF")
            return None  # Return None on error

        # Save to in-memory stream and return object
    print("Generating GIF object for display...")
    try:
        gif_stream = io.BytesIO()
        frames[0].save(
            gif_stream,
            format="GIF",  # MUST specify format for stream
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop,
            optimize=False,
        )
        gif_bytes = gif_stream.getvalue()
        print("GIF object generated.")

        if IPYTHON_AVAILABLE:
            # If IPython is available, return the displayable Image object
            print("Returning IPython.display.Image object.")
            return IPImage(data=gif_bytes)
        # Otherwise, return the raw bytes
        print("IPython not found. Returning raw GIF bytes.")
        return gif_bytes
    except Exception:
        logging.exception("Error generating GIF")
        return None  # Return None on error


# --- Example Usage (Updated) ---
if __name__ == "__main__":
    if GridBoard is None:
        print("Cannot run example because GridBoard class import failed.")
    else:
        print("Running example: Creating GIF with player circle and wall text.")

        # 1. Create trajectory
        trajectory = []
        board = GridBoard(size=6)  # Slightly larger board

        # Add boundary (REQUIRED due to render_np bug!)
        boundary_mask_arr = np.zeros((6, 6), dtype=np.uint8)
        boundary_mask_arr[0, :] = 1
        boundary_mask_arr[-1, :] = 1
        boundary_mask_arr[:, 0] = 1
        boundary_mask_arr[:, -1] = 1
        board.addMask("boundary", boundary_mask_arr, "#")

        # Add a Wall component
        board.addPiece("Wall", "W", (2, 2))
        board.addPiece(
            "Wall", "W", (2, 3)
        )  # Need unique names if using dict keys directly!
        # Let's assume Wall component can occupy multiple cells or use a mask
        wall_mask_arr = np.zeros((6, 6), dtype=np.uint8)
        wall_mask_arr[2, 2:4] = 1
        board.addMask("Walls", wall_mask_arr, "W")  # Use a mask for multiple walls

        board.addPiece("Player", "P", (1, 1))
        board.addPiece("Goal", "+", (4, 4))
        board.addPiece("Pit", "-", (1, 4))
        trajectory.append(board)

        # --- Simulate moves (create new board states for trajectory) ---
        # Move 1
        board_state_1 = GridBoard(size=6)
        board_state_1.addMask("boundary", boundary_mask_arr, "#")
        board_state_1.addMask("Walls", wall_mask_arr, "W")  # Add wall mask
        board_state_1.addPiece("Player", "P", (1, 2))
        board_state_1.addPiece("Goal", "+", (4, 4))
        board_state_1.addPiece("Pit", "-", (1, 4))
        trajectory.append(board_state_1)

        # Move 2 (hits wall - assuming move logic prevents this or state repeats)
        # Let's assume player moves down instead
        board_state_2 = GridBoard(size=6)
        board_state_2.addMask("boundary", boundary_mask_arr, "#")
        board_state_2.addMask("Walls", wall_mask_arr, "W")
        board_state_2.addPiece("Player", "P", (2, 1))
        board_state_2.addPiece("Goal", "+", (4, 4))
        board_state_2.addPiece("Pit", "-", (1, 4))
        trajectory.append(board_state_2)  # Player below wall

        # Move 3
        board_state_3 = GridBoard(size=6)
        board_state_3.addMask("boundary", boundary_mask_arr, "#")
        board_state_3.addMask("Walls", wall_mask_arr, "W")
        board_state_3.addPiece("Player", "P", (3, 1))
        board_state_3.addPiece("Goal", "+", (4, 4))
        board_state_3.addPiece("Pit", "-", (1, 4))
        trajectory.append(board_state_3)

        # Move 4: Towards Goal
        board_state_4 = GridBoard(size=6)
        board_state_4.addMask("boundary", boundary_mask_arr, "#")
        board_state_4.addMask("Walls", wall_mask_arr, "W")
        board_state_4.addPiece("Player", "P", (4, 1))
        board_state_4.addPiece("Goal", "+", (4, 4))
        board_state_4.addPiece("Pit", "-", (1, 4))
        trajectory.append(board_state_4)

        # Move 5: Reach Goal
        board_state_5 = GridBoard(size=6)
        board_state_5.addMask("boundary", boundary_mask_arr, "#")
        board_state_5.addMask("Walls", wall_mask_arr, "W")
        board_state_5.addPiece("Player", "P", (4, 4))
        board_state_5.addPiece("Goal", "+", (4, 4))
        board_state_5.addPiece("Pit", "-", (1, 4))
        trajectory.append(board_state_5)

        # 2. Define colors, including Wall
        my_colors = {
            "Player": (30, 144, 255),
            "Goal": (0, 255, 0),
            "Pit": (255, 0, 0),
            "boundary": (64, 64, 64),
            "Walls": (169, 169, 169),  # DarkGray for Wall background
        }

        # Define text colors
        my_player_text_color = (255, 255, 255)  # White text on Blue Player
        my_wall_text_color = (0, 0, 0)  # Black text on Gray Wall

        # 3. Generate single image
        print("\nGenerating single image (last state) with circle, text, grid:")
        # Try to find a common system font (adjust path if needed)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if not Path(font_path).exists():
            font_path = None  # Fallback to default if not found

        last_state_img = render_color(
            trajectory[-1],
            color_map=my_colors,
            cell_size=50,
            show_grid=True,
            grid_color=(0, 0, 0),
            grid_width=1,
            player_text="P",
            player_text_color=my_player_text_color,
            wall_text="W",
            wall_text_color=my_wall_text_color,
            font_path=font_path,  # Pass font path
        )
        try:
            last_state_img.save("grid_last_state_final.png")
            print("Saved last state as grid_last_state_final.png")
        except Exception as e:
            print(f"Error saving PNG: {e}")

    # 4. Create and Save the GIF to a file (Optional)
    print("\nGenerating GIF with circle, text, grid (Saving to File):")
    create_gif(  # Returns None when filename is provided
        board_states=trajectory,
        filename="example_trajectory_final.gif",  # Provide filename
        duration=600,
        color_map=my_colors,
        cell_size=50,
        show_grid=True,
        grid_color=(200, 200, 200),
        grid_width=1,
        player_text="P",
        player_text_color=my_player_text_color,
        wall_text="W",
        wall_text_color=my_wall_text_color,
        font_path=font_path,
    )

    # 5. Generate GIF object for display (e.g., in Jupyter)
    print("\nGenerating GIF object for display (e.g., in Jupyter):")
    gif_object = create_gif(  # Call *without* filename
        board_states=trajectory,
        filename=None,  # Explicitly None, or just omit it
        duration=600,
        color_map=my_colors,
        cell_size=50,
        show_grid=True,
        grid_color=(200, 200, 200),
        grid_width=1,
        player_text="P",
        player_text_color=my_player_text_color,
        wall_text="W",
        wall_text_color=my_wall_text_color,
        font_path=font_path,
    )

    # How to display in Jupyter:
    # If you run this script and gif_object is returned,
    # in a subsequent Jupyter cell, simply having `gif_object` as the last line
    # will display it (if IPython is available).
    # Or explicitly:
    if gif_object and IPYTHON_AVAILABLE:
        print("\nTo display the GIF object in Jupyter, use:")
        print("from IPython.display import display")
        print("display(gif_object)")
        # display(gif_object) # Uncomment this line if running directly in Jupyter
    elif gif_object and not IPYTHON_AVAILABLE:
        print(
            "\nGIF generated as raw bytes (IPython not available). Length:",
            len(gif_object),
        )
        # You could save these bytes manually:
        # with open("manual_save.gif", "wb") as f:
        #     f.write(gif_object)
