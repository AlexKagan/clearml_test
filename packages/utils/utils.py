import os
import shutil
import json
import numpy as np
import pandas as pd
import pickle
import glob
import datetime
import copy


class IO:
    @classmethod
    def create_dir(cls, path):
        if os.path.exists(path):
            return ("Trying to create folder %s, but path already exists" % path)
        os.mkdir(path)
        if os.path.exists(path):
            return ("Created folder %s" %path)
        else:
            return ("Failed to create folder %s" %path)

    @classmethod
    def delete_file(cls, path):
        try:
            os.remove(path)
        except OSError as e:
            "Failed to delete file. Error: {}".format(e)

    @classmethod
    def remove_dir(cls, path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            print("Directory not removed. Error: {}".format(e))
            return "Directory not removed. Error: {}".format(e)

    @classmethod
    def copy_file(cls, path_to_file, path_to_copy):
        try:
            shutil.copy(path_to_file, path_to_copy)
        except OSError as e:
            "Failed copy file. Error: {}".format(e)

    @classmethod
    def copy_dir(cls, path_to_dir, path_to_copy):
        try:
            shutil.copytree(path_to_dir, path_to_copy)
        except OSError as e:
            print('Directory not copied. Error: {}'.format(e))
            return 'Directory not copied. Error: {}'.format(e)

    @classmethod
    def read_json(cls, path):
        try:
            with open(path, 'r') as fout:
                json_file = json.load(fout)
            return json_file
        except OSError as e:
            return "Failed to read file. Error: {}".format(e)


    @classmethod
    def write_json(cls, file, path):
        try:
            with open(path, 'w') as fout:
                json.dump(file, fout, indent=4)
        except OSError as e:
            return "Failed to write json. Error: {}".format(e)

    @classmethod
    def write_csv(cls, file, path):
        if isinstance(file, pd.DataFrame):
            try:
                file.to_csv(path, index=False)
            except OSError as e:
                print('Failed to write csv. Error: {}'.format(e))

    @classmethod
    def write_pickle(cls, file, path):
        try:
            with open(path, "wb") as fout:
                pickle.dump(file, fout)
        except OSError as e:
            return 'Failed to write pickle. Error: {}'.format(e)

    @classmethod
    def read_pickle(cls, path):
        try:
            with open(path, "rb") as fout:
                return pickle.load(fout)
        except OSError as e:
            return 'Failed to read pickle file. Error: {}'.format(e)

    @classmethod
    def read_text(cls, path):
        try:
            with open(path, 'r') as fout:
                return fout.read()
        except OSError as e:
            return 'Failed to read text. Error: {}'.format(e)

    @classmethod
    def write_text(cls, path, line):
        try:
            with open(path, 'w') as fout:
                fout.write(line)
        except OSError as e:
            return 'Failed to write text. Error: {}'.format(e)
