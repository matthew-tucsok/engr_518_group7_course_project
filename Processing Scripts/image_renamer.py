import os
import glob


def main():
    image_id = 500
    cwd = os.getcwd()
    print(cwd)
    os.chdir(cwd)
    files = glob.glob(cwd + '/**/*.png', recursive=True)

    for file in files:
        os.rename(file, cwd+'/output/nf_' + str(image_id) + '.png')
        image_id += 1


if __name__ == '__main__':
    main()
