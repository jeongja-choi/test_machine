import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="GPT-4.0-mini 시스템 프롬프트 스코어 채점기",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사용자 정의 CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .score-high {
        color: #28a745;
        font-weight: bold;
    }
    .score-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .score-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🎯 GPT-4.0-mini 시스템 프롬프트 스코어 채점기</h1>
    <p>CSV 파일을 업로드하여 시스템 프롬프트의 품질을 평가하고 라벨을 생성하세요</p>
</div>
""", unsafe_allow_html=True)

class PromptScorer:
    def __init__(self):
        self.scoring_criteria = {
            'accuracy': 0.90,     # 정확도 90%
            'length': 0.10        # 길이 10%
        }
        self.max_length = 3000    # 최대 글자 수
        
    def calculate_accuracy_score(self, text):
        """정확도 점수 계산 (90% 가중치)"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0
            
        score = 50
        
        # 1. 명확성 (25점)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
            if 20 <= avg_sentence_length <= 100:
                score += 6
        
        clear_instructions = ['다음', '아래', '위의', '이것을', '그것을', '해주세요', '하십시오', '생성', '분석', '작성']
        instruction_count = sum(1 for word in clear_instructions if word in text)
        score += min(6, instruction_count * 2)
        
        words = text.split()
        if words:
            unique_words = set(words)
            repetition_ratio = len(unique_words) / len(words)
            if repetition_ratio > 0.7:
                score += 6
        
        # 2. 구체성 (25점)
        if re.search(r'\d+', text):
            score += 6
        
        example_keywords = ['예를 들어', '예시', '다음과 같이', '구체적으로', '예:', '예제']
        example_count = sum(1 for word in example_keywords if word in text)
        score += min(8, example_count * 2)
        
        technical_terms = ['API', '데이터', '알고리즘', '모델', '시스템', '프로세스', '분석', '처리', '생성', '평가']
        technical_count = sum(1 for term in technical_terms if term in text)
        score += min(6, technical_count * 1)
        
        # 3. 완성도 (25점)
        text_length = len(text)
        if 50 <= text_length <= 500:
            score += 8
        elif text_length < 20:
            score -= 8
        
        has_introduction = bool(re.search(r'^(안녕|시작|먼저|우선|당신은)', text.strip()))
        has_conclusion = bool(re.search(r'(마지막|결론|끝|완료|해주세요|하십시오)$', text.strip()))
        
        if has_introduction:
            score += 4
        if has_conclusion:
            score += 4
        
        has_steps = bool(re.search(r'\d+\.|첫째|둘째|셋째|단계|절차', text))
        if has_steps:
            score += 4
        
        # 4. 효과성 (25점)
        action_words = ['생성', '분석', '처리', '실행', '수행', '작성', '검토', '평가', '계산', '추천']
        action_count = sum(1 for word in action_words if word in text)
        score += min(8, action_count * 2)
        
        goal_keywords = ['목표', '목적', '결과', '달성', '완성', '해결']
        goal_count = sum(1 for word in goal_keywords if word in text)
        if goal_count > 0:
            score += 6
        
        constraint_keywords = ['단', '하지만', '제외', '제한', '조건', '규칙', '주의']
        constraint_count = sum(1 for word in constraint_keywords if word in text)
        if constraint_count > 0:
            score += 6
        
        return max(0, min(100, score))
    
    def calculate_length_score(self, text):
        """길이 점수 계산 (10% 가중치)"""
        if not isinstance(text, str):
            return 0
            
        text_length = len(text)
        
        # 3000자 초과 시 0점
        if text_length > self.max_length:
            return 0
        
        # 최적 길이 범위: 100-1500자
        if 100 <= text_length <= 1500:
            return 100
        elif 50 <= text_length < 100:
            return 80
        elif 1500 < text_length <= 2500:
            return 70
        elif 2500 < text_length <= 3000:
            return 50
        elif text_length < 50:
            return 30
        else:
            return 0
    
    def generate_analysis_content(self, text, accuracy_score, length_score, total_score):
        """분석 내용 생성"""
        if not isinstance(text, str):
            return "분석 불가: 유효하지 않은 텍스트"
        
        text_length = len(text)
        analysis_parts = []
        
        # 길이 분석
        if text_length > self.max_length:
            analysis_parts.append(f"⚠️ 글자 수 초과 ({text_length}/{self.max_length}자)")
        elif text_length < 50:
            analysis_parts.append(f"📝 매우 짧은 프롬프트 ({text_length}자)")
        elif 100 <= text_length <= 1500:
            analysis_parts.append(f"✅ 적절한 길이 ({text_length}자)")
        else:
            analysis_parts.append(f"📏 길이: {text_length}자")
        
        # 정확도 분석
        if accuracy_score >= 80:
            analysis_parts.append("🎯 높은 정확도")
        elif accuracy_score >= 60:
            analysis_parts.append("📊 보통 정확도")
        else:
            analysis_parts.append("⚡ 낮은 정확도")
        
        # 구조 분석
        has_clear_instruction = any(word in text for word in ['해주세요', '하십시오', '생성', '분석', '작성'])
        has_examples = any(word in text for word in ['예를 들어', '예시', '예:', '예제'])
        has_constraints = any(word in text for word in ['단', '제한', '조건', '규칙'])
        
        structure_elements = []
        if has_clear_instruction:
            structure_elements.append("명확한 지시")
        if has_examples:
            structure_elements.append("예시 포함")
        if has_constraints:
            structure_elements.append("제약 조건")
        
        if structure_elements:
            analysis_parts.append(f"🔧 구조: {', '.join(structure_elements)}")
        
        # 전체 평가
        if total_score >= 80:
            analysis_parts.append("⭐ 우수한 프롬프트")
        elif total_score >= 60:
            analysis_parts.append("👍 양호한 프롬프트")
        else:
            analysis_parts.append("🔄 개선 필요")
        
        return " | ".join(analysis_parts)
    
    def analyze_high_quality_patterns(self, texts, labels):
        """고품질 프롬프트의 공통 패턴 분석"""
        high_quality_texts = [text for text, label in zip(texts, labels) if label == 1]
        
        if not high_quality_texts:
            return {
                'common_keywords': [],
                'pattern_counts': {},
                'total_high_quality': 0,
                'sample_texts': []
            }
        
        # 키워드 추출
        all_words = []
        for text in high_quality_texts:
            if isinstance(text, str):
                words = re.findall(r'[가-힣a-zA-Z]+', text)
                all_words.extend([word for word in words if len(word) > 1])
        
        # 빈도 분석
        word_freq = Counter(all_words)
        common_keywords = word_freq.most_common(20)
        
        # 패턴 분석
        patterns = {
            '명령어': ['생성', '분석', '작성', '처리', '수행', '실행', '검토', '평가', '해주세요', '하십시오'],
            '구조어': ['다음', '아래', '위의', '단계', '절차', '방법', '과정', '순서대로', '체계적으로'],
            '예시어': ['예를 들어', '예시', '구체적으로', '예:', '예제', '다음과 같이'],
            '제약어': ['단', '하지만', '제외', '제한', '조건', '규칙', '주의', '제한사항'],
            '목표어': ['목표', '목적', '결과', '달성', '완성', '해결', '결과물']
        }
        
        pattern_counts = {}
        for category, keywords in patterns.items():
            count = 0
            for text in high_quality_texts:
                if isinstance(text, str):
                    for keyword in keywords:
                        if keyword in text:
                            count += 1
                            break  # 텍스트당 한 번만 카운트
            pattern_counts[category] = count
        
        return {
            'common_keywords': common_keywords,
            'pattern_counts': pattern_counts,
            'total_high_quality': len(high_quality_texts),
            'sample_texts': high_quality_texts[:3]  # 상위 3개 샘플
        }
    
    def get_accuracy_keywords(self):
        """정확도 100점을 위한 추천 키워드"""
        return {
            '필수 명령어': ['생성해주세요', '분석해주세요', '작성해주세요', '처리해주세요'],
            '구조화 키워드': ['다음과 같이', '아래 단계로', '순서대로', '체계적으로'],
            '구체성 키워드': ['구체적으로', '상세하게', '예를 들어', '다음 예시처럼'],
            '제약 조건': ['단,', '하지만', '제한사항:', '주의사항:', '조건:'],
            '목표 명시': ['목표는', '목적은', '결과물은', '달성하고자 하는'],
            '역할 정의': ['당신은 전문가입니다', '당신의 역할은', '전문적인', '숙련된']
        }
    
    def validate_text_length(self, text):
        """텍스트 길이 검증"""
        if not isinstance(text, str):
            return False, "유효하지 않은 텍스트입니다."
        
        text_length = len(text)
        if text_length > self.max_length:
            return False, f"텍스트가 너무 깁니다. ({text_length}/{self.max_length}자)"
        
        return True, "유효한 길이입니다."
    
    def calculate_total_score(self, text):
        """총 점수 계산 (정확도 90% + 길이 10%)"""
        # 길이 검증
        is_valid, message = self.validate_text_length(text)
        if not is_valid:
            return {
                'total_score': 0,
                'accuracy_score': 0,
                'length_score': 0,
                'label': 0,
                'analysis': f"❌ {message}",
                'is_valid': False
            }
        
        accuracy_score = self.calculate_accuracy_score(text)
        length_score = self.calculate_length_score(text)
        
        total_score = (
            accuracy_score * self.scoring_criteria['accuracy'] +
            length_score * self.scoring_criteria['length']
        )
        
        analysis = self.generate_analysis_content(text, accuracy_score, length_score, total_score)
        
        return {
            'total_score': round(total_score, 2),
            'accuracy_score': accuracy_score,
            'length_score': length_score,
            'label': 1 if total_score >= 70 else 0,
            'analysis': analysis,
            'is_valid': True
        }

def load_data():
    """데이터 로드 함수"""
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드하세요",
        type=['csv'],
        help="프롬프트 데이터가 포함된 CSV 파일을 선택하세요"
    )
    
    if uploaded_file is not None:
        try:
            # CSV 파일 읽기
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            
            # 파일 정보 표시
            file_size = uploaded_file.size
            st.success(f"✅ 파일이 성공적으로 업로드되었습니다!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 행 수", f"{len(df):,}")
            with col2:
                st.metric("총 열 수", len(df.columns))
            with col3:
                st.metric("파일 크기", f"{file_size/1024:.1f} KB")
            with col4:
                st.metric("메모리 사용량", f"{df.memory_usage(deep=True).sum()/1024:.1f} KB")
            
            return df
            
        except Exception as e:
            st.error(f"❌ 파일을 읽는 중 오류가 발생했습니다: {str(e)}")
            return None
    
    return None

def display_data_info(df):
    """데이터 정보 표시"""
    st.subheader("📊 데이터 정보")
    
    # 기본 정보
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**데이터 형태:**")
        st.write(f"- 행: {len(df):,}")
        st.write(f"- 열: {len(df.columns)}")
        st.write(f"- 메모리 사용량: {df.memory_usage(deep=True).sum()/1024:.1f} KB")
    
    with col2:
        st.write("**열 정보:**")
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            st.write(f"- {col}: {dtype} (결측값: {null_count})")
    
    # 전체 데이터 표시
    st.write("**전체 데이터:**")
    st.dataframe(df, use_container_width=True)

def analyze_single_prompt(scorer):
    """단일 프롬프트 분석"""
    st.subheader("✍️ 프롬프트 직접 입력 및 분석")
    
    # 정확도 100점을 위한 키워드 추천
    with st.expander("💡 정확도 100점을 위한 추천 키워드", expanded=False):
        accuracy_keywords = scorer.get_accuracy_keywords()
        
        cols = st.columns(3)
        for i, (category, keywords) in enumerate(accuracy_keywords.items()):
            with cols[i % 3]:
                st.write(f"**{category}:**")
                for keyword in keywords:
                    if st.button(f"📝 {keyword}", key=f"keyword_{category}_{keyword}"):
                        # 키워드를 텍스트 영역에 추가하는 기능은 streamlit 제한으로 인해 구현 어려움
                        st.info(f"'{keyword}' 키워드를 프롬프트에 포함해보세요!")
        
        st.markdown("""
        **🎯 고품질 프롬프트 작성 팁:**
        - 명확한 역할 정의로 시작하세요 (예: "당신은 전문적인 데이터 분석가입니다")
        - 구체적인 작업 지시를 포함하세요 (예: "다음 데이터를 분석해주세요")
        - 예시나 구체적인 설명을 추가하세요 (예: "예를 들어", "구체적으로")
        - 제약 조건이나 주의사항을 명시하세요 (예: "단,", "주의사항:")
        - 최종 목표나 결과물을 명확히 하세요 (예: "목표는", "결과물은")
        - 적절한 길이 유지 (100-1500자 권장)
        """)
    
    # 프롬프트 입력
    user_prompt = st.text_area(
        "분석할 프롬프트를 입력하세요:",
        height=200,
        max_chars=3000,
        help="최대 3000자까지 입력 가능합니다. 위의 추천 키워드를 참고하여 작성해보세요!"
    )
    
    # 글자 수 표시
    char_count = len(user_prompt)
    if char_count > 3000:
        st.error(f"❌ 글자 수 초과: {char_count}/3000자")
    elif char_count > 2500:
        st.warning(f"⚠️ 글자 수 주의: {char_count}/3000자")
    else:
        st.info(f"📝 현재 글자 수: {char_count}/3000자")
    
    if st.button("🎯 프롬프트 분석하기", type="primary", disabled=len(user_prompt.strip()) == 0):
        if user_prompt.strip():
            with st.spinner("프롬프트를 분석하고 있습니다..."):
                score_result = scorer.calculate_total_score(user_prompt)
                
                # 결과 표시
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("총점", f"{score_result['total_score']:.1f}점")
                with col2:
                    st.metric("정확도 점수", f"{score_result['accuracy_score']:.1f}점")
                with col3:
                    st.metric("길이 점수", f"{score_result['length_score']:.1f}점")
                with col4:
                    label_text = "고품질 (1)" if score_result['label'] == 1 else "저품질 (0)"
                    label_color = "🟢" if score_result['label'] == 1 else "🔴"
                    st.metric("라벨", f"{label_color} {label_text}")
                
                # 분석 내용 표시
                st.subheader("📊 상세 분석")
                st.info(score_result['analysis'])
                
                # 개선 제안
                if score_result['accuracy_score'] < 100:
                    st.subheader("🚀 정확도 향상을 위한 제안")
                    
                    suggestions = []
                    
                    # 역할 정의 확인
                    if not any(word in user_prompt for word in ['당신은', '전문가', '전문적인']):
                        suggestions.append("• 명확한 역할 정의 추가 (예: '당신은 전문적인 ~입니다')")
                    
                    # 명령어 확인
                    if not any(word in user_prompt for word in ['해주세요', '하십시오', '생성', '분석', '작성']):
                        suggestions.append("• 구체적인 명령어 추가 (예: '~해주세요', '분석해주세요')")
                    
                    # 예시 확인
                    if not any(word in user_prompt for word in ['예를 들어', '예시', '구체적으로']):
                        suggestions.append("• 예시나 구체적인 설명 추가 (예: '예를 들어', '구체적으로')")
                    
                    # 구조 확인
                    if not any(word in user_prompt for word in ['다음', '단계', '순서']):
                        suggestions.append("• 구조화된 지시사항 추가 (예: '다음 단계로', '순서대로')")
                    
                    # 목표 확인
                    if not any(word in user_prompt for word in ['목표', '목적', '결과']):
                        suggestions.append("• 목표나 결과물 명시 (예: '목표는', '결과물은')")
                    
                    if suggestions:
                        for suggestion in suggestions:
                            st.write(suggestion)
                    else:
                        st.success("프롬프트가 이미 잘 구성되어 있습니다!")
                
                # 점수 시각화
                fig = go.Figure(go.Bar(
                    x=['정확도 (90%)', '길이 (10%)', '총점'],
                    y=[score_result['accuracy_score'], score_result['length_score'], score_result['total_score']],
                    marker_color=['#4facfe', '#00f2fe', '#667eea']
                ))
                fig.update_layout(
                    title="점수 분석",
                    yaxis_title="점수",
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig, use_container_width=True)

def analyze_prompts(df, scorer):
    """프롬프트 분석 및 스코어링"""
    st.subheader("🎯 CSV 프롬프트 스코어링")
    
    # 텍스트 컬럼 선택
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not text_columns:
        st.error("❌ 텍스트 컬럼을 찾을 수 없습니다.")
        return None
    
    # 컬럼 선택 방식 선택
    col_selection_type = st.radio(
        "분석 방식을 선택하세요:",
        ["단일 컬럼", "복합 컬럼 (제목+내용)"],
        help="단일 컬럼: 하나의 컬럼만 분석 / 복합 컬럼: 여러 컬럼을 결합하여 분석"
    )
    
    if col_selection_type == "단일 컬럼":
        selected_column = st.selectbox(
            "분석할 프롬프트 컬럼을 선택하세요:",
            text_columns,
            help="시스템 프롬프트가 포함된 컬럼을 선택하세요"
        )
        combine_columns = False
        selected_columns = [selected_column]
    else:
        st.write("**복합 분석을 위한 컬럼들을 선택하세요:**")
        selected_columns = st.multiselect(
            "분석할 컬럼들을 선택하세요 (예: 제목, 내용, 설명 등):",
            text_columns,
            help="선택된 컬럼들의 내용이 결합되어 분석됩니다"
        )
        
        if not selected_columns:
            st.warning("⚠️ 최소 하나의 컬럼을 선택해주세요.")
            return None
        
        combine_columns = True
        selected_column = " + ".join(selected_columns)  # 표시용
    
    if st.button("🚀 스코어 계산 시작", type="primary"):
        with st.spinner("프롬프트를 분석하고 있습니다..."):
            # 원본 데이터프레임 복사
            result_df = df.copy()
            
            # 스코어 계산을 위한 리스트
            labels = []
            total_scores = []
            accuracy_scores = []
            length_scores = []
            analyses = []
            
            progress_bar = st.progress(0)
            total_rows = len(df)
            
            for idx, row in df.iterrows():
                # 텍스트 결합 처리
                if combine_columns:
                    text_parts = []
                    for col in selected_columns:
                        if pd.notna(row[col]):
                            text_parts.append(str(row[col]))
                    text = " ".join(text_parts)
                else:
                    text = str(row[selected_columns[0]]) if pd.notna(row[selected_columns[0]]) else ""
                
                score_result = scorer.calculate_total_score(text)
                
                labels.append(score_result['label'])
                total_scores.append(score_result['total_score'])
                accuracy_scores.append(score_result['accuracy_score'])
                length_scores.append(score_result['length_score'])
                analyses.append(score_result['analysis'])
                
                progress_bar.progress((idx + 1) / total_rows)
            
            # 원본 데이터프레임에 새 컬럼들 추가
            result_df['label'] = labels  # 0과 1로만 출력되는 라벨 컬럼
            result_df['total_score'] = total_scores
            result_df['accuracy_score'] = accuracy_scores
            result_df['length_score'] = length_scores
            result_df['analysis'] = analyses
            
            # 결과 표시
            st.subheader("📊 분석 완료된 전체 데이터")
            
            # 통계 정보
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_score = result_df['total_score'].mean()
                st.metric("평균 점수", f"{avg_score:.1f}점")
            with col2:
                high_quality = (result_df['label'] == 1).sum()
                st.metric("고품질 프롬프트", f"{high_quality}개")
            with col3:
                low_quality = (result_df['label'] == 0).sum()
                st.metric("저품질 프롬프트", f"{low_quality}개")
            with col4:
                quality_ratio = (high_quality / len(result_df)) * 100
                st.metric("품질 비율", f"{quality_ratio:.1f}%")
            
            # 전체 데이터 테이블 표시 (라벨 컬럼 포함)
            st.write("**분석 결과가 포함된 전체 데이터:**")
            
            # 라벨 컬럼 강조 표시를 위한 스타일링
            def highlight_labels(val):
                if val == 1:
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                elif val == 0:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                return ''
            
            styled_df = result_df.style.applymap(highlight_labels, subset=['label'])
            st.dataframe(styled_df, use_container_width=True)
            
            # 고품질 프롬프트 패턴 분석
            st.subheader("🔍 고품질 프롬프트 패턴 분석")
            
            # 텍스트 추출 (복합 컬럼 고려)
            texts = []
            for idx, row in df.iterrows():
                if combine_columns:
                    text_parts = []
                    for col in selected_columns:
                        if pd.notna(row[col]):
                            text_parts.append(str(row[col]))
                    text = " ".join(text_parts)
                else:
                    text = str(row[selected_columns[0]]) if pd.notna(row[selected_columns[0]]) else ""
                texts.append(text)
            
            pattern_analysis = scorer.analyze_high_quality_patterns(texts, labels)
            
            st.write(f"**분석 대상:** 총 {len(labels)}개 프롬프트 중 고품질(라벨=1) {sum(labels)}개")
            st.write(f"**분석 컬럼:** {selected_column}")
            
            if pattern_analysis and pattern_analysis.get('total_high_quality', 0) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🎯 라벨=1을 위한 공통 키워드 TOP 10:**")
                    if pattern_analysis['common_keywords']:
                        keywords_df = pd.DataFrame(
                            pattern_analysis['common_keywords'][:10], 
                            columns=['키워드', '빈도']
                        )
                        st.dataframe(keywords_df, use_container_width=True)
                    
                    st.write("**📊 패턴별 사용 빈도:**")
                    pattern_df = pd.DataFrame(
                        list(pattern_analysis['pattern_counts'].items()),
                        columns=['패턴 유형', '사용 횟수']
                    )
                    st.dataframe(pattern_df, use_container_width=True)
                    
                    # 패턴 분석 상세 결과
                    st.write("**🔍 패턴 분석 상세 결과:**")
                    total_high = pattern_analysis['total_high_quality']
                    for pattern_type, count in pattern_analysis['pattern_counts'].items():
                        percentage = (count / total_high * 100) if total_high > 0 else 0
                        st.write(f"- {pattern_type}: {count}/{total_high}개 ({percentage:.1f}%)")
                
                with col2:
                    st.write("**✨ 고품질 프롬프트 샘플:**")
                    for i, sample in enumerate(pattern_analysis['sample_texts'], 1):
                        with st.expander(f"샘플 {i} (점수: {[s for s, l in zip(total_scores, labels) if l == 1][i-1] if i <= len([s for s, l in zip(total_scores, labels) if l == 1]) else 'N/A'}점)"):
                            st.write(sample[:500] + "..." if len(sample) > 500 else sample)
                    
                    # 패턴 시각화
                    if pattern_analysis['pattern_counts']:
                        fig_pattern = px.bar(
                            x=list(pattern_analysis['pattern_counts'].keys()),
                            y=list(pattern_analysis['pattern_counts'].values()),
                            title="패턴별 사용 빈도",
                            labels={'x': '패턴 유형', 'y': '사용 횟수'},
                            color=list(pattern_analysis['pattern_counts'].values()),
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig_pattern, use_container_width=True)
                
                # 정확도 100점에 가까운 대안 제시
                st.subheader("🚀 정확도 100점 달성을 위한 맞춤형 대안")
                
                # 현재 고품질 프롬프트들의 평균 점수 계산
                high_quality_scores = [score for score, label in zip(total_scores, labels) if label == 1]
                if high_quality_scores:
                    avg_high_score = sum(high_quality_scores) / len(high_quality_scores)
                    max_score = max(high_quality_scores)
                    
                    col_alt1, col_alt2 = st.columns(2)
                    
                    with col_alt1:
                        st.write("**📈 현재 성과 분석:**")
                        st.write(f"- 고품질 프롬프트 평균 점수: {avg_high_score:.1f}점")
                        st.write(f"- 최고 점수: {max_score:.1f}점")
                        st.write(f"- 100점까지 필요한 개선: {100 - max_score:.1f}점")
                        
                        # 부족한 패턴 분석
                        missing_patterns = []
                        for pattern_type, count in pattern_analysis['pattern_counts'].items():
                            coverage = (count / total_high * 100) if total_high > 0 else 0
                            if coverage < 80:  # 80% 미만 사용률
                                missing_patterns.append((pattern_type, coverage))
                        
                        if missing_patterns:
                            st.write("**⚠️ 개선이 필요한 패턴:**")
                            for pattern, coverage in missing_patterns:
                                st.write(f"- {pattern}: {coverage:.1f}% 사용률")
                    
                    with col_alt2:
                        st.write("**💡 100점 달성을 위한 구체적 제안:**")
                        
                        suggestions = [
                            "🎯 **역할 정의 강화**: '당신은 [분야]의 전문가로서 [구체적 역할]을 수행합니다'",
                            "📝 **명령어 구체화**: '다음 작업을 단계별로 수행해주세요: 1) ... 2) ... 3) ...'",
                            "📋 **예시 추가**: '예를 들어, [구체적 예시]와 같이 작성해주세요'",
                            "⚖️ **제약 조건 명시**: '단, 다음 조건을 반드시 준수해주세요: [조건1], [조건2]'",
                            "🎯 **목표 명확화**: '최종 결과물은 [구체적 형태]이며, [평가 기준]을 만족해야 합니다'",
                            "🔄 **피드백 요청**: '작업 완료 후 [검증 방법]으로 확인해주세요'"
                        ]
                        
                        for suggestion in suggestions:
                            st.write(suggestion)
                        
                        # 최적화된 프롬프트 템플릿 제공
                        st.write("**📋 100점 프롬프트 템플릿:**")
                        template = """
                        **역할**: 당신은 [전문 분야]의 숙련된 전문가입니다.
                        
                        **작업**: 다음 작업을 체계적으로 수행해주세요:
                        1. [구체적 작업 1]
                        2. [구체적 작업 2] 
                        3. [구체적 작업 3]
                        
                        **예시**: 예를 들어, [구체적 예시]와 같은 형태로 작성해주세요.
                        
                        **제약사항**: 
                        - [제약조건 1]
                        - [제약조건 2]
                        
                        **목표**: 최종 결과물은 [구체적 형태]이며, [품질 기준]을 만족해야 합니다.
                        
                        **확인**: 작업 완료 후 [검증 방법]으로 품질을 확인해주세요.
                        """
                        st.code(template, language="text")
                else:
                    st.info("고품질 프롬프트가 없어 맞춤형 대안을 제시할 수 없습니다.")
            else:
                st.warning("고품질 프롬프트(라벨=1)가 없어 패턴 분석을 수행할 수 없습니다.")
                st.info("💡 팁: 더 나은 프롬프트 작성을 위해 '직접 입력' 탭의 추천 키워드를 참고하세요!")
                
                # 고품질 프롬프트가 없을 때의 일반적인 개선 제안
                st.subheader("🚀 프롬프트 품질 향상을 위한 일반 가이드")
                
                col_guide1, col_guide2 = st.columns(2)
                
                with col_guide1:
                    st.write("**📝 기본 구조 개선:**")
                    basic_tips = [
                        "명확한 역할 정의 (당신은 ~입니다)",
                        "구체적인 작업 지시 (~해주세요)", 
                        "단계별 절차 제시 (1, 2, 3단계)",
                        "예시나 샘플 제공 (예를 들어)",
                        "제약 조건 명시 (단, ~조건)"
                    ]
                    for tip in basic_tips:
                        st.write(f"• {tip}")
                
                with col_guide2:
                    st.write("**🎯 고급 최적화 기법:**")
                    advanced_tips = [
                        "목표와 결과물 명확화",
                        "품질 기준 및 평가 방법 제시",
                        "컨텍스트와 배경 정보 제공",
                        "피드백 및 수정 요청 포함",
                        "적절한 길이 유지 (100-1500자)"
                    ]
                    for tip in advanced_tips:
                        st.write(f"• {tip}")
            
            # 다운로드 버튼
            csv_data = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 분석 결과 다운로드 (CSV)",
                data=csv_data,
                file_name="prompt_analysis_results.csv",
                mime="text/csv"
            )
            
            # 스코어만 포함된 데이터프레임도 반환 (기존 함수와의 호환성을 위해)
            scores_data = []
            for idx, row in result_df.iterrows():
                # 복합 컬럼인 경우 텍스트 재구성
                if combine_columns:
                    text_parts = []
                    for col in selected_columns:
                        if pd.notna(row[col]):
                            text_parts.append(str(row[col]))
                    display_text = " ".join(text_parts)
                else:
                    display_text = str(row[selected_columns[0]]) if pd.notna(row[selected_columns[0]]) else ""
                
                scores_data.append({
                    'index': idx,
                    'prompt': display_text[:100] + "..." if len(display_text) > 100 else display_text,
                    'total_score': row['total_score'],
                    'accuracy_score': row['accuracy_score'],
                    'length_score': row['length_score'],
                    'label': row['label'],
                    'analysis': row['analysis']
                })
            
            scores_df = pd.DataFrame(scores_data)
            return scores_df
    
    return None

def display_results(scores_df, original_df):
    """결과 표시"""
    st.subheader("📈 분석 결과")
    
    # 전체 통계
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = scores_df['total_score'].mean()
        st.metric("평균 점수", f"{avg_score:.1f}점")
    
    with col2:
        high_quality = (scores_df['label'] == 1).sum()
        st.metric("고품질 프롬프트", f"{high_quality}개")
    
    with col3:
        low_quality = (scores_df['label'] == 0).sum()
        st.metric("저품질 프롬프트", f"{low_quality}개")
    
    with col4:
        quality_ratio = (high_quality / len(scores_df)) * 100
        st.metric("품질 비율", f"{quality_ratio:.1f}%")
    
    # 점수 분포 시각화
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            scores_df, 
            x='total_score', 
            nbins=20,
            title="점수 분포",
            labels={'total_score': '총 점수', 'count': '빈도'}
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        label_counts = scores_df['label'].value_counts()
        fig_pie = px.pie(
            values=label_counts.values,
            names=['저품질 (0)', '고품질 (1)'],
            title="품질 라벨 분포"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # 세부 점수 분석
    st.subheader("🔍 세부 점수 분석")
    
    criteria_scores = scores_df[['clarity', 'specificity', 'completeness', 'effectiveness']].mean()
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=criteria_scores.values,
        theta=['명확성', '구체성', '완성도', '효과성'],
        fill='toself',
        name='평균 점수'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="평가 기준별 평균 점수"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # 결과 테이블
    st.subheader("📋 상세 결과")
    
    # 점수에 따른 색상 적용
    def color_score(val):
        if val >= 80:
            return 'background-color: #d4edda; color: #155724'
        elif val >= 60:
            return 'background-color: #fff3cd; color: #856404'
        else:
            return 'background-color: #f8d7da; color: #721c24'
    
    # 라벨에 따른 색상 적용
    def color_label(val):
        if val == 1:
            return 'background-color: #28a745; color: white; font-weight: bold'
        else:
            return 'background-color: #dc3545; color: white; font-weight: bold'
    
    styled_df = scores_df.style.applymap(
        color_score, 
        subset=['total_score', 'accuracy_score', 'length_score']
    ).applymap(
        color_label,
        subset=['label']
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # 다운로드 버튼
    csv_data = scores_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 결과 다운로드 (CSV)",
        data=csv_data,
        file_name="prompt_scores.csv",
        mime="text/csv"
    )

def main():
    """메인 함수"""
    # 사이드바
    st.sidebar.header("⚙️ 설정")
    st.sidebar.markdown("---")
    
    # 스코어러 초기화
    scorer = PromptScorer()
    
    # 평가 기준 설정
    st.sidebar.subheader("📊 평가 기준 가중치")
    accuracy_weight = st.sidebar.slider("정확도", 0.0, 1.0, 0.90, 0.05)
    length_weight = st.sidebar.slider("길이", 0.0, 1.0, 0.10, 0.05)
    
    # 가중치 정규화
    total_weight = accuracy_weight + length_weight
    if total_weight > 0:
        scorer.scoring_criteria = {
            'accuracy': accuracy_weight / total_weight,
            'length': length_weight / total_weight
        }
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 라벨 임계값")
    threshold = st.sidebar.slider("고품질 판정 점수", 50, 90, 70, 5)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📏 글자 수 제한")
    max_chars = st.sidebar.number_input("최대 글자 수", min_value=1000, max_value=5000, value=3000, step=100)
    scorer.max_length = max_chars
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **사용 방법:**
    1. 직접 프롬프트를 입력하거나
    2. CSV 파일을 업로드하세요
    3. 프롬프트 컬럼을 선택하세요
    4. 스코어 계산을 시작하세요
    5. 결과를 확인하고 다운로드하세요
    
    **라벨 설명:**
    - 1: 고품질 프롬프트
    - 0: 저품질 프롬프트
    
    **채점 기준:**
    - 정확도: 90% (명확성, 구체성, 완성도, 효과성)
    - 길이: 10% (3000자 이내 기준)
    """)
    
    # 메인 컨텐츠 - 탭으로 구성
    tab1, tab2, tab3, tab4 = st.tabs(["✍️ 직접 입력", "📁 CSV 업로드", "📊 데이터 정보", "📈 통계 분석"])
    
    with tab1:
        # 직접 프롬프트 입력 및 분석
        analyze_single_prompt(scorer)
    
    with tab2:
        # CSV 파일 업로드 및 분석
        df = load_data()
        
        if df is not None:
            # 업로드된 데이터 즉시 표시
            st.subheader("📋 업로드된 원본 데이터")
            st.dataframe(df, use_container_width=True)
            
            scores_df = analyze_prompts(df, scorer)
            
            # 세션 상태에 저장
            if scores_df is not None:
                st.session_state['scores_df'] = scores_df
                st.session_state['original_df'] = df
    
    with tab3:
        # 데이터 정보 표시
        if 'original_df' in st.session_state:
            display_data_info(st.session_state['original_df'])
        else:
            st.info("먼저 CSV 파일을 업로드해주세요.")
    
    with tab4:
        # 통계 분석
        if 'scores_df' in st.session_state:
            st.subheader("📊 고급 통계 분석")
            scores_df = st.session_state['scores_df']
            
            # 상관관계 분석
            if 'accuracy_score' in scores_df.columns and 'length_score' in scores_df.columns:
                corr_data = scores_df[['accuracy_score', 'length_score', 'total_score']].corr()
                
                fig_corr = px.imshow(
                    corr_data,
                    text_auto=True,
                    aspect="auto",
                    title="평가 기준 간 상관관계",
                    labels=dict(x="평가 기준", y="평가 기준", color="상관계수")
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # 박스 플롯
                melted_df = scores_df.melt(
                    id_vars=['label'],
                    value_vars=['accuracy_score', 'length_score'],
                    var_name='평가기준',
                    value_name='점수'
                )
                melted_df['평가기준'] = melted_df['평가기준'].map({
                    'accuracy_score': '정확도',
                    'length_score': '길이'
                })
                
                fig_box = px.box(
                    melted_df,
                    x='평가기준',
                    y='점수',
                    color='label',
                    title="라벨별 평가 기준 점수 분포",
                    labels={'label': '라벨', 'points': '점수'}
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
                # 산점도
                fig_scatter = px.scatter(
                    scores_df,
                    x='accuracy_score',
                    y='length_score',
                    color='label',
                    size='total_score',
                    title="정확도 vs 길이 점수 분포",
                    labels={'accuracy_score': '정확도 점수', 'length_score': '길이 점수', 'label': '라벨'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("먼저 프롬프트 분석을 수행해주세요.")

if __name__ == "__main__":
    main()
