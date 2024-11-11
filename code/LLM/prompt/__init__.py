"""
프롬프트 템플릿을 버전별로 관리하고 제공하는 모듈입니다.

## 주요 기능
- relabel: 텍스트 레이블 수정을 위한 프롬프트 생성
- denoise: 텍스트 노이즈 제거를 위한 프롬프트 생성  
- augment: 텍스트 증강을 위한 프롬프트 생성

## 사용 방법
version 파라미터를 통해 원하는 버전의 프롬프트 템플릿을 선택할 수 있습니다.

## 모델
- LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct에서 가장 잘 동작합니다.

Examples:   
>>> prompt = get_denoise_prompt(1, "noisy text") 
>>> prompt = get_relabel_prompt(1, "sample text", "target")
>>> prompt = get_agument_prompt(1, "original text")
"""

from .agument import (
    get_agument_system_message,
    get_prompt_agument,
    get_prompt_synonyms,
    get_prompt_title_to_article,
    get_prompt_article_to_title,
)
from .denoise import get_prompt_denoise, get_system_message_denoise
from .relabel import get_prompt_relabel, get_system_messaget_relabel
