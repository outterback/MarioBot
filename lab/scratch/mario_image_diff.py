from pathlib import Path
import cv2

def main():
    dir = Path('/home/oscar/Dev/PycharmProjects/serpent/datasets/collect_frames/w1-2')
    files = sorted(dir.glob('*.png'))
    win_1 = "mario1"
    win_2 = "mario2"
    win_3 = "mario3"

    cv2.namedWindow(win_1)
    cv2.namedWindow(win_2)
    cv2.namedWindow(win_3)

    #cv2.moveWindow(win_2, x=400, y=0)
    #cv2.moveWindow(win_3, x=800, y=0)
    f_idx = 0
    while True:
        f_t = files[f_idx]
        f_tp1 = files[f_idx + 1]
        img_t = cv2.imread(str(f_t))
        img_tp1 = cv2.imread(str(f_tp1))

        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
        img_tp1 = cv2.cvtColor(img_tp1, cv2.COLOR_BGR2GRAY)
        ret, result_img = cv2.threshold(img_tp1 - img_t, 1, 255, cv2.THRESH_BINARY)
        result_img = cv2.resize(result_img, (0, 0), fx=0.2, fy=0.2)

        cv2.imshow(win_1, img_t)
        cv2.imshow(win_2, img_tp1)
        cv2.imshow(win_3, result_img)
        print(f'Sum of diff_values: {sum(sum(result_img))}')
        pressed_key = cv2.waitKey(50)
        f_idx += 1
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('a'):
            f_idx -= 1
        elif pressed_key == ord('d'):
            f_idx += 1

        if f_idx < 0:
            f_idx = 0
        if f_idx >= len(files) - 2:
            f_idx = 0

    pass


if __name__ == '__main__':
    print('==== WELCOME ====')
    main()
