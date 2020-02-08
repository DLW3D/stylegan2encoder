import cv2
import sys
import os.path


def f2f(cascade_file, origin_dir, face_dir):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
    cascade = cv2.CascadeClassifier(cascade_file)

    image_names = [files for root, dirs, files in os.walk(origin_dir)][0]
    print('find %s files in %s' % (len(image_names), origin_dir))
    for index, image_name in enumerate(image_names):
        image_path = os.path.join(origin_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print('error read image: %s\n' % image_name)
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(50, 50))
        i = 0
        for (x, y, w, h) in faces:
            # cropped = image[y: y + h, x: x + w]  # origin
            # cropped = image[int(y * 0.25): y + h, int(x * 0.90): x + int(w * 1.25)]  # gwern
            top = max(y - int(h * 0.5), 0)
            left = max(x - int(w * 0.25), 0)
            cropped = image[top: top + int(h * 1.5), left: left + int(w * 1.5)]  # math
            cv2.imwrite(os.path.join(face_dir, "%s_%s.png" % (image_name, str(i))), cropped)
            i += 1
        print("\r", "%s/%s " % (index, len(image_names)), end="", flush=True)

    print('Done!')


if len(sys.argv) != 4:
    sys.stderr.write("usage: crop.py <animeface.xml file>  <input folder> <output folder>\n")
    sys.exit(-1)

f2f(sys.argv[1], sys.argv[2], sys.argv[3])
