from database_genertation import main
import os

def safecreate(path):
    if not os.path.exists(path):
        os.mkdir(path)

def line1():
    imgpath = '/home/zhaojin/data/TacomaBridge/capture/high-reso-clip2/frame20190505_150247.316.png'
    labelpath = '/home/zhaojin/Nutstore Files/Nutstore/Annotation/frame20190505_150247.316/frame00002.npz'

    outpath = '/home/zhaojin/data/TacomaBridge/segdata/train4'


    outimg = os.path.join(outpath, 'img')
    outlbl = os.path.join(outpath, 'mask')

    safecreate(outpath)
    safecreate(outimg)
    safecreate(outlbl)


    lumi = 10
    noise = 7
    datanum = 2000

    print('start generate dataset')
    main(['--img', imgpath, '--label', labelpath, '--output', outpath, '--noise', str(noise), '--lumination', str(lumi), '-d', str(datanum)])

def line2():
    imgpath = '/home/zhaojin/Nutstore Files/Nutstore/Annotation/frame20190505_150250.960/frame20190505_150250.960.png'
    labelpath = '/home/zhaojin/Nutstore Files/Nutstore/Annotation/frame20190505_150250.960/frame00067.npz'

    outpath = '/home/zhaojin/data/TacomaBridge/segdata/train5'


    outimg = os.path.join(outpath, 'img')
    outlbl = os.path.join(outpath, 'mask')

    safecreate(outpath)
    safecreate(outimg)
    safecreate(outlbl)


    lumi = 10
    noise = 7
    datanum = 2000

    print('start generate dataset')
    main(['--img', imgpath, '--label', labelpath, '--output', outpath, '--noise', str(noise), '--lumination', str(lumi), '-d', str(datanum)])

if __name__ == "__main__":
    line1()
    line2()