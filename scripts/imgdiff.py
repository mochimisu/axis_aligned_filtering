from PIL import Image
import sys,os,shutil,fnmatch,glob,re

if len(sys.argv) < 4:
  raise RuntimeError("Not enough arguments; input1 input2 output magnification")
  sys.exit()

img_dir1 = sys.argv[1]
img_dir2 = sys.argv[2]
diff_name = sys.argv[3]
mag = float(sys.argv[4])

img1 = Image.open(img_dir1)
img2 = Image.open(img_dir2)

if img1.size != img2.size:
  raise RuntimeError("inconsistent dimensions")

diff = Image.new(img1.mode, img1.size)

#load pixel arrays
img1_pix = img1.load()
img2_pix = img2.load()
diff_pix = diff.load()

for i in range(img1.size[0]):
  for j in range(img2.size[1]):
    diff_pix[i,j] = (int(mag*abs(img1_pix[i,j][0] - img2_pix[i,j][0])),
      int(mag*abs(img1_pix[i,j][1] - img2_pix[i,j][1])),
      int(mag*abs(img1_pix[i,j][2] - img2_pix[i,j][2])))

#save
diff.save(diff_name)
