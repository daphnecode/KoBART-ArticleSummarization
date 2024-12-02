import streamlit as st
import re
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from model import KoBARTConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import nltk
# nltk.download('punkt') #처음 수행할 때만 필요.

@st.cache_resource()
def load_model():
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
    model = KoBARTConditionalGeneration.load_from_checkpoint("YourModelPath.ckpt")
    return tokenizer, model

def load_model_for_test():
    model1 = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
    tokenizer1 = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
    return model1, tokenizer1



# 입력된 텍스트를 요약 생성하기 위해 모델에 전달하고 요약 결과를 반환함
def test_generate_summary(text):
    input_ids = tokenizer1.encode(text, return_tensors="pt")  # 입력 텍스트를 토크나이저로 인코딩하여 텐서로 변환
    output = model1.generate(input_ids, eos_token_id=1, max_length=1024, num_beams=5)  # 생성 파라미터 설정 후 요약 생성
    summary = tokenizer1.decode(output[0], skip_special_tokens=True)  # 생성된 텍스트를 디코딩하여 요약을 반환
    return summary

def evaluate_summary_quality(original_text, generated_summary):
    # 원문과 요약문의 단어 수를 계산
    original_word_count = len(original_text.split())  # 원문 단어 수 계산
    summary_word_count = len(generated_summary.split())  # 요약문 단어 수 계산
    compression_ratio = (1 - (summary_word_count / original_word_count)) * 100  # 요약 압축률 계산

    # 원문과 요약문에서 각 단어의 집합을 추출하여 공통 키워드의 비율을 계산
    original_keywords = set(re.findall(r'\b\w+\b', original_text.lower()))  # 원문에서 키워드 추출
    summary_keywords = set(re.findall(r'\b\w+\b', generated_summary.lower()))  # 요약문에서 키워드 추출
    common_keywords = original_keywords.intersection(summary_keywords)  # 원문과 요약문 간의 공통된 키워드 추출
    keyword_coverage_ratio = (len(common_keywords) / len(original_keywords)) * 100  # 주요 키워드 포함률 계산

    return original_word_count, summary_word_count, compression_ratio, keyword_coverage_ratio

# TF-IDF를 사용하여 원문과 요약문 간의 코사인 유사도 계산
def compute_similarity_score(original_text, generated_summary):
    vectorizer = TfidfVectorizer().fit_transform([original_text, generated_summary])  # TF-IDF 벡터화
    vectors = vectorizer.toarray()  # 벡터를 배열로 변환
    similarity_score = cosine_similarity(vectors)[0, 1]  # 코사인 유사도를 계산하여 반환
    return similarity_score


def generate_summary(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output_ids = model.model.generate( 
        input_ids,
        max_length=128,               # 최대 생성 길이
        min_length=30,                # 최소 생성 길이
        num_beams=5,                  # Beam Search의 빔 수
        repetition_penalty=1.2,       # 반복 패널티
        top_k=50,                     # Top-k 샘플링
        early_stopping=True,          # 조기 종료 조건
    ) # 정확성
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def split_text_into_parts(text, num_parts=3):
    sentences = nltk.sent_tokenize(text)
    parts = []
    start_idx = 0
    
    if len(sentences) <= num_parts:
        parts.append(text)
        return parts
    
    avg_sentences_per_part = len(sentences) // num_parts
    remainder = len(sentences) % num_parts  
    
    for i in range(num_parts):
        end_idx = start_idx + avg_sentences_per_part + (1 if i < remainder else 0)  
        parts.append(' '.join(sentences[start_idx:end_idx]))
        start_idx = end_idx
        
    return parts


model1, tokenizer1 = load_model_for_test()
tokenizer, model = load_model()

st.title("KoBART 요약 Test")
text = st.text_area("뉴스 입력:")

st.markdown("## 뉴스 원문")
st.write(text)

if text:
    if text.strip() == "":
        st.error("입력 텍스트가 비어있습니다.")
    else:
        with st.spinner('processing..'):
            text = text.replace('\n', '')

            summary_text1 = test_generate_summary(text)

            parts = split_text_into_parts(text)
            _text = ""
            for part in parts:
                _text += generate_summary(part)
            summary_text2 = generate_summary(text)

        st.markdown("## KoBART 요약 결과")
        st.write("\ndigit82/kobart-summarization  : ", summary_text1)
        st.write("\n학습한 모델  : ", summary_text2)

        # 요약 품질 평가 출력
        original_word_count1, summary_word_count1, compression_ratio1, keyword_coverage_ratio1 = evaluate_summary_quality(text, summary_text1)
        st.write("\ndigit82/kobart-summarization 요약 품질 평가:")
        st.write(f"- 원문 길이: {original_word_count1} 단어")
        st.write(f"- 요약 길이: {summary_word_count1} 단어")
        st.write(f"- 요약 압축률: {compression_ratio1:.2f}%")
        st.write(f"- 주요 키워드 포함률: {keyword_coverage_ratio1:.2f}%")

        original_word_count2, summary_word_count2, compression_ratio2, keyword_coverage_ratio2 = evaluate_summary_quality(text, summary_text2)
        st.write("\n학습한 모델 요약 품질 평가:")
        st.write(f"- 원문 길이: {original_word_count2} 단어")
        st.write(f"- 요약 길이: {summary_word_count2} 단어")
        st.write(f"- 요약 압축률: {compression_ratio2:.2f}%")
        st.write(f"- 주요 키워드 포함률: {keyword_coverage_ratio2:.2f}%")

        # 유사도 계산 및 출력
        similarity_score = compute_similarity_score(summary_text1, summary_text2)
        st.write(f"\n\n두 요약문 간의 유사도: {similarity_score:.2f}")
        
        