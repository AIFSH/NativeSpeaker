import argparse
from src.core import Core


def main(args):
    assert args.input_file_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')) , "video file is expected"
    assert args.output_file_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')), "output file should be defined"
    core = Core(args)
    core()



if __name__ == "__main__":
    langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'ja','hu','ko',"hi"]
    parser = argparse.ArgumentParser(description="Make your charater speak native language!")
    parser.add_argument("-i","--input_file_path", help="path to video file")
    parser.add_argument("-l","--lang_code",help="native language code you want speak",choices=list(langs))
    parser.add_argument("-o","--output_file_path",help="define your output filename")
    parser.add_argument("-ht","--hf_token",help="hf token for diarizing speaker")
    parser.add_argument("-mn","--model_name",help="model for wav2lip", choices=["wav2lip","wav2lip_gan"], default="wav2lip")
    parser.add_argument("-xn","--xt_version_name",help="version name of xtts ", default="v2.0.3")
    args = parser.parse_args()

    main(args)