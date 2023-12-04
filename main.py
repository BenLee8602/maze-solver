import sys
import cv2

from src import preprocess, solve


# maze solver pipeline
def solve_maze(image):
    bin = preprocess.binary_image(image)
    corners = preprocess.detect_maze(image)
    maze = preprocess.perspective_transform_maze(bin, corners)
    
    contours, kernel = solve.get_maze_contours(maze)
    path = solve.get_maze_path(contours, kernel)
    out = solve.build_output(image, path)

    return out


# solve maze in the given file(s)
for filename in sys.argv[1:]:
    image = cv2.imread(filename)
    if image is None:
        print(f"unable to load {filename}")
        continue
    sol = solve_maze(image)
    cv2.imshow(f"maze solver ({filename})", sol if sol is not None else image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# use video capture if no files given
if len(sys.argv) > 1:
    exit(0)

vid = cv2.VideoCapture(0)
while True:
    _, frame = vid.read()
    sol = solve_maze(frame)
    cv2.imshow("maze solver", sol if sol is not None else frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
