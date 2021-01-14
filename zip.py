from zipfile import ZipFile
import os
from os.path import basename
from optparse import OptionParser


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file_path")
    parser.add_option("-z", "--ziped_file", dest="ziped_file_path")
    opts,args = parser.parse_args()
    
    with ZipFile(opts.ziped_file_path, 'w') as zipObj:
        for folderName, subfolders, filenames in os.walk(opts.file_path):
            for filename in filenames:
                filePath = os.path.join(folderName, filename)
                zipObj.write(filePath, basename(filePath))