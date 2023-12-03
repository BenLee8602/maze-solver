import sys
import cv2

from src import preprocess


# maze solving pipeline
def solve_maze(image):
    # preprocess
    bin = preprocess.binary_image(image)
    corners = preprocess.detect_maze(bin)
    if not corners:
        return None
    maze = preprocess.perspective_transform_maze(bin, corners)

    # solve
    pass

    # display
    pass

    return maze # temp


# solve maze in the given file(s)
for filename in sys.argv[1:]:
    image = cv2.imread(filename)
    if not image:
        print(f"unable to load {filename}")
        continue
    sol = solve_maze(image)
    cv2.imshow(f"maze solver ({filename})", sol if sol else image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# use video capture if no files given
if len(sys.argv) > 1:
    exit(0)

vid = cv2.VideoCapture(0)
while True:
    _, frame = vid.read()
    sol = solve_maze(frame)
    cv2.imshow("maze solver", sol if sol else frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
