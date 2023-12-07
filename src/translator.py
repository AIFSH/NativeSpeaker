from typing import Any
import translators as ts
from src.log_helper import HandleLog
logger = HandleLog()

class Translator:

    def __init__(self,work_with_human=False) -> None:
        # _ = ts.preaccelerate_and_speedtest()
        self.work_with_human = work_with_human

    def __call__(self,text,from_lang,to_lang,*args: Any, **kwds: Any) -> Any:
        assert from_lang != to_lang,"same lang code error,translator only work in language to another language"
        if self.work_with_human:
            lience = input("!!!注意，出现这个提示是因为您自行修改了相关代码,请不要做偏离原文内容的手工翻译，否则后果自负，与该项目开源作者无关！我已经阅读并同意该声明。\n(!!!Attention!This prompt appears because you modified the code yourself,Please do not deviate from the original content of manual translation, or bear the consequences,It has nothing to do with the author of this project! I have read and agree with the statement)\t yes | no:\n").strip()
            if "y" not in lience:
                self.work_with_human = False
            
        if "zh" in to_lang:
            to_lang = "zh"
        logger.info(f"{from_lang} {to_lang} {text} ")
        try:
            dst_text = ts.translate_text(query_text=text,translator="qqTranSmart",
                                         from_language=from_lang,to_language=to_lang)
        except ts.server.TranslatorError:
            dst_text = input("translator failed,input by self:")
            dst_text = dst_text.strip()
            return dst_text
        logger.info("dst_text:{}".format(dst_text))
        if self.work_with_human:
            if_by_hand = input("translate by hand? 1 by hand, 0 pass:\t")
            if if_by_hand == "1":
                dst_text = input("input by hand:\n").strip()
                logger.info(f"dst_text edited:{dst_text}")

        return dst_text