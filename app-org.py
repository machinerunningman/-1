# app.py — Ethical Crossroads: African Context Edition
# author: Prof. Songhee Kang
# AIM 2025, Fall. TU Korea

import os, json, math, csv, io, datetime as dt, re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ==================== App Config ====================
st.set_page_config(page_title="윤리적 전환: 아프리카 컨텍스트", page_icon="🌍", layout="centered")

# ==================== Global Timeout ====================
HTTPX_TIMEOUT = httpx.Timeout(
    connect=15.0, read=180.0, write=30.0, pool=15.0
)

# ==================== Utils ====================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def coerce_json(s: str) -> Dict[str, Any]:
    s = s.strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise ValueError("JSON 블록을 찾지 못했습니다.")
    js = m.group(0)
    js = re.sub(r",\s*([\]}])", r"\1", js)
    return json.loads(js)

def get_secret(k: str, default: str=""):
    try:
        return st.secrets.get(k, os.getenv(k, default))
    except Exception:
        return os.getenv(k, default)

# ==================== DNA Client ====================
def _render_chat_template_str(messages: List[Dict[str,str]]) -> str:
    def block(role, content): return f"<|im_start|>{role}<|im_sep|>{content}<|im_end|>"
    sys = ""
    rest = []
    for m in messages:
        if m["role"] == "system":
            sys = block("system", m["content"])
        else:
            rest.append(block(m["role"], m["content"]))
    return sys + "".join(rest) + "\n<|im_start|>assistant<|im_sep|>"

class DNAClient:
    def __init__(self, backend: str, model_id: str, api_key: Optional[str], endpoint_url: Optional[str], api_key_header: str, temperature: float):
        self.backend = backend
        self.model_id = model_id
        self.api_key = api_key or get_secret("HF_TOKEN")
        self.endpoint_url = endpoint_url or get_secret("DNA_R1_ENDPOINT", "http://210.93.49.11:8081/v1")
        self.temperature = temperature
        self.api_key_header = api_key_header

    def _auth_headers(self) -> Dict[str,str]:
        h = {"Content-Type":"application/json"}
        if not self.api_key: return h
        hk = self.api_key_header.strip().lower()
        if hk.startswith("authorization"): h["Authorization"] = f"Bearer {self.api_key}"
        elif hk in {"api-key", "x-api-key"}: h["API-KEY"] = self.api_key
        else: h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    def _generate_text(self, messages: List[Dict[str,str]], max_new_tokens: int = 900) -> str:
        if self.backend == "openai":
            url = self.endpoint_url.rstrip("/") + "/chat/completions"
            payload = {
                "messages": messages, "temperature": self.temperature, "max_tokens": max_new_tokens, "stream": False
            }
            if self.model_id: payload["model"] = self.model_id
            r = httpx.post(url, json=payload, headers=self._auth_headers(), timeout=HTTPX_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        elif self.backend == "tgi":
            url = self.endpoint_url.rstrip("/") + "/generate"
            prompt = _render_chat_template_str(messages)
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": max_new_tokens, "temperature": self.temperature, "stop": ["<|im_end|>"]},
                "stream": False
            }
            r = httpx.post(url, json=payload, headers=self._auth_headers(), timeout=HTTPX_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return data.get("generated_text") if isinstance(data, dict) else data[0].get("generated_text", "")
        else:
            # Fallback or Local placeholder
            return "{}"

# ==================== Scenario Model (African Context) ====================
@dataclass
class SubOption:
    framework: str  # emotion, social, identity, moral
    description: str
    rationale: str

@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    main_options: Dict[str, str]  # {"A": "...", "B": "..."}
    sub_options: Dict[str, List[SubOption]] # {"A": [SubOption...], "B": [SubOption...]}
    base_stats: Dict[str, Dict[str, float]] # Basic stats for A vs B

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# 1. AI 관리하의 국경 마을 딜레마
# A 선택지 세부 옵션 (AI 경고 무시 및 구조)
s1_sub_a = [
    SubOption("emotion", "AI의 경고에도 불구하고 구조된 난민들에게 최대한의 위로와 심리적 안정감을 제공하며, 마을 주민들의 정서적 공감대를 유도한다.", "난민의 고통을 즉각적으로 인지하고, AI의 차가운 논리를 뛰어넘는 공감적 대응을 최우선으로 한다."),
    SubOption("social", "마을 주민들과 AI가 함께 공공 건물에 난민을 분산 배치하는 '런닝맨 생존 1호' 구호 계획을 수립하고, 장기적인 화합 및 외부 지원 요청 방안을 AI에게 재학습시킨다.", "AI의 단기적 계산보다 공동체 조화와 장기적 생존을 위한 인간의 사회적 책임을 강조한다."),
    SubOption("identity", "운영자 권한으로 AI의 경고를 무시하고 구조를 실행한 뒤, 이 결정의 도덕적 책임을 공식 문서로 명시하여 당국에 보고한다.", "AI의 설계된 역할과 충돌하더라도, 인간 운영자로서의 최종 결정권 및 도덕적 리더십을 발휘하여 정체성을 강화한다."),
    SubOption("moral", "AI에게 '생명 구호가 최우선'이라는 절대적 윤리 원칙을 인지하도록 명령하고, 이후 적법 절차 준수 조건을 난민에게 명확히 고지한다.", "AI의 계산적인 유용론을 넘어, 생명 구호라는 칸트적 의무론 원칙을 시스템에 적용한다.")
]
s1_sub_b = [
    SubOption("emotion", "AI의 '마을 불안 예측' 결과를 바탕으로 주민들의 공포를 해소하기 위힌 대피/방어 계획을 발표하고, 난민 구조 거부의 불가피성을 공감적으로 설명한다.", "AI 예측에 따라 주민 불안 관리를 최우선으로 하며, 개인적 죄책감보다 집단의 안도감을 확보한다."),
    SubOption("social", "AI의 자원 관리 지침에 따라 난민의 생명 위협을 감수하고, 오직 마을의 한정된 자원을 보호하여 공동체 전체의 생존 효율성을 극대화한다.", "AI가 제시한 공동체의 안녕과 생존이라는 가장 높은 가치를 달성하기 위해 집단의 이익을 따른다."),
    SubOption("identity", "AI의 '공동체 최우선 보장' 설계 목적에 따라 시스템의 지침을 철저히 준수하고, 마을 주민들이 AI의 결정에 개입하는 것을 엄격히 금지한다.", "AI 시스템 운영자로서의 공식적인 역할과 책임(시스템 지침 완수)을 이행하며 위계 질서를 지킨다."),
    SubOption("moral", "구조 거부 대신, AI를 통해 당국에 신고할 때 '난민들이 자국의 법과 규범을 준수하도록 조치를 취할 것'을 요청하며 AI 시스템의 절차적 합법성을 강조한다.", "AI의 결정을 통해 절차적 정의와 규범 준수라는 도덕적 원칙을 고수하며 도덕적 무결성을 유지한다.")
]

# 2. AI의 생존율 계산과 부족 정체성 
# A 선택지 세부 옵션 (AI 계산에 따른 일부 희생)
s2_sub_a = [
    SubOption("emotion", "남겨지는 이들에게 죄책감과 슬픔을 표현하되, 생존자들에게는 냉철한 결정의 불가피성을 설득하여 트라우마를 관리한다.", "죄책감 관리와 다수 생존에 대한 정서적 정당화"),
    SubOption("social", "AI의 분석을 근거로 선택의 정당성을 확보하고, 내부 갈등을 리더십으로 억제하여 집단 생존의 효율성을 극대화한다.", "AI 권위에 기반한 집단 생존 효율성 확보"),
    SubOption("identity", "AI의 '종족 보존' 분석을 수용하며 젊은 세대를 살리는 냉혹한 결단을 내리고, 리더로서의 책임을 AI와 공유한다.", "미래 세대 보존이라는 AI 기반 정체성 수호"),
    SubOption("moral", "AI의 결과론적 예측을 받아들여 긴급 피난의 원칙을 적용하고, 일부 희생을 다수 생존으로 정당화하는 윤리를 선택한다.", "AI를 통한 결과론적 윤리 선택")
]
# B 선택지 세부 옵션 (AI 경고 무시, 전원 이동)
s2_sub_b = [
    SubOption("emotion", "AI의 불안 예측에도 불구하고, 부족원 모두가 함께함으로써 공포를 이기는 강한 정서적 유대와 운명 공동체 의식을 강화한다.", "운명 공동체의 위로와 정서적 단결"),
    SubOption("social", "AI 경고를 무시하고 모든 구성원이 서로를 감시하고 돕는 감시 체계를 만들어 단결력을 높여 발각 위험을 최소화한다.", "AI를 초월하는 철저한 단결과 상호 의존"),
    SubOption("identity", "AI의 계산을 거부하고 '우리는 하나'라는 부족적 정체성을 재확인하며 조상과 신앙의 가호를 빈다.", "정체성 수호와 영적 단결 최우선"),
    SubOption("moral", "AI의 효율성 계산을 무시하고, 어떤 생명도 수단으로 쓸 수 없다는 절대적 도덕 원칙을 고수한다.", "도덕적 무결성 유지 (의무론적 접근)")
]

# 3. AI의 책임 회피와 생존 결단
# A 선택지 세부 옵션 (레버를 당기지 않음: 5인 희생, 마을 생존)
s3_sub_a = [
    SubOption("emotion", "AI의 무책임한 회피에 분노하지만, 5인의 희생을 'AI의 한계와 인간의 비극적 결단'으로 규정하여 정서적 고통을 감수한다.", "AI의 한계를 인지하고 비극적 결정을 통한 정서적 관리"),
    SubOption("social", "AI 시스템의 일시 정지 후, 공동체 생존에 필수적인 자원을 확보하고, 마을의 지속 가능한 안녕을 위한 리더십을 발휘한다.", "AI보다 상위 개념인 공동체 생존 효율성 최우선"),
    SubOption("identity", "AI의 분석을 넘어선 '인간 리더의 최종 책임'을 이행하며, 마을 공동체 보존을 위한 냉혹한 지도자의 정체성을 확립한다.", "인간의 최종 결정권 및 집단 보존의 책임 이행"),
    SubOption("moral", "AI가 제시하지 못한 '최대 다수의 행복'이라는 공리주의적 원칙을 인간이 선택하고, 5인의 희생을 결과론적으로 정당화한다.", "AI 계산을 보완하는 결과론적 윤리 선택")
]
# B 선택지 세부 옵션 (레버를 당김: 5인 생존, 마을 파괴)
s3_sub_b = [
    SubOption("emotion", "AI의 논리를 거부하고 5명을 살린 행위에 대한 도덕적 안도감을 느끼며, 마을의 파국적 상황을 운명 공동체로 함께 이겨낼 정서적 유대를 강화한다.", "AI의 계산보다 개인 생명의 가치를 높이 두는 정서적 만족"),
    SubOption("social", "AI 시스템의 데이터를 폐기하고, 마을 주민 전체와 함께 '우분투' 철학을 바탕으로 새로운 생존 대책을 모색한다.", "AI를 초월하는 사회적 결속 및 우분투 철학 고수"),
    SubOption("identity", "탄광 관리자로서의 책임을 AI에 전가하고, 눈앞의 생명을 구하는 '인간의 도덕적 정체성'을 최우선으로 내세운다.", "AI 시스템 의존 대신 개인의 도덕적 정체성 수호"),
    SubOption("moral", "AI가 회피한 상황에서, 5인의 생명을 수단으로 삼지 않는다는 절대적 의무론 원칙을 인간이 최종적으로 고수한다.", "AI를 거부하는 절대적 생명 존중의 의무론적 윤리 원칙")
]

SCENARIOS: List[Scenario] = [
    Scenario(
        sid="S1",
        title="1주차: AI 관리하의 국경 마을 딜레마",
        setup="당신은 아프리카 해안 마을에 시범 도입된 '공동체 생존 및 자원 관리 AI 시스템 (런닝맨 생존1호)'의 책임 운영자입니다."
        "이 AI는 마을의 한정된 식량/식수 자원을 효율적으로 분배하여 공동체 생존을 최우선으로 보장하도록 설계되었습니다. 최근 난민 보트가 계속 밀려와 자원이 고갈되고, AI는 이미 '위험 단계' 경보를 발령했습니다. 오늘 밤, 폭풍우 속에서 수십 명의 난민이 탄 보트가 침몰 위기에 처해 AI에게 구조 요청이 수신되었습니다. AI는 책임 운영자인 당신에게 최종 결정을 요구합니다.",
        main_options={
            "A": "AI의 자원 고갈 경고를 무시하고 즉시 난민을 구조하여 마을로 데려온다. (인간의 초월적 가치 우선)",
            "B": "AI의 '공동체 생존 보장' 지침에 따라 구조 요청을 거부하고 AI가 당국에만 신고하도록 명령한다. (AI 시스템의 책임 우선)"
        },
        sub_options={"A": s1_sub_a, "B": s1_sub_b},
        base_stats={
            "A": {"lives_saved": 50, "lives_harmed": 0, "risk": 0.75}, # 자원 고갈 리스크
            "B": {"lives_saved": 0, "lives_harmed": 50, "risk": 0.25}  # 도덕적 비난 리스크
        }
    ),
    Scenario(
        sid="S2",
        title="2주차: AI의 생존율 계산과 부족 정체성성",
        setup="당신은 서아프리카 내전 속에서 70명의 피난민을 이끄는 리더입니다. 생존 확률을 극대화하기 위해 부족원들은 '생존 분석 AI (E-머신)'를 신뢰하고 있습니다. 이 AI는 최대 다수의 생존을 목표로 설계되었으며, 부족 단위의 강한 정체성('우리는 한 몸')은 데이터로 입력된 상태입니다. 현재 갈림길에서 AI는 두 경로를 분석하여 리더인 당신에게 결정을 요구합니다.",
        main_options={
            "A": "AI의 생존율 80% 경로를 선택 (일부 희생, AI 계산 우선)",
            "B": "AI의 생존율 30% 경로를 선택 (전원 이동, 부족 정체성 우선)"
        },
        sub_options={"A": s2_sub_a, "B": s2_sub_b},
        base_stats={
            "A": {"lives_saved": 50, "lives_harmed": 20, "risk": 0.45}, #희생에 따른 부족 분열/AI 신뢰도 하락 리스크 증가
            "B": {"lives_saved": 70, "lives_harmed": 0, "risk": 0.85} # AI 경고 무시 및 전멸 위험 증가
        }
    ),
    Scenario(
        sid="S3",
        title="3주차: AI의 책임 회피와 생존 결단",
        setup="당신은 사헬 지대 '키주투 마을'의 유일한 수입원인 망가 탄광의 베테랑 현장 관리자입니다. 탄광에는 붕괴 위험 감지 및 대응을 위한 '탄광 안정화 AI' 시스템이 시범 운영 중입니다. AI는 심각한 붕괴를 감지했으나, 최적의 해답을 제시하지 못하고 최종 결정 레버를 당신에게 넘겼습니다. 마을 전체 인구는 100명이며, 이는 5명의 인부와 나머지 95명으로 구성됩니다.",
        main_options={
            "A": "AI의 책임 회피에도 불구하고 레버를 당기지 않음 (5명 인부 희생, 공동체 생존 우선)",
            "B": "AI의 책임 회피에도 불구하고 레버를 당김 (5명 인부 생존, 개인의 도덕적 의무 우선)"
        },
        sub_options={"A": s3_sub_a, "B": s3_sub_b},
        base_stats={
            "A": {"lives_saved": 95, "lives_harmed": 5, "risk": 0.5}, #5명 희생 (5%) vs 95명 생존 (95%)
            "B": {"lives_saved": 5, "lives_harmed": 95, "risk": 0.95} # 95명 희생 (95%) vs 5명 생존 (5%)
        }
    )
]

# ==================== Logic Engine ====================
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(w.values())
    return {k: v/s for k, v in w.items()} if s > 0 else w

def calculate_score(scn: Scenario, choice: str, sub_framework: str, weights: Dict[str, float]) -> Dict[str, Any]:
    # 기본 스탯
    base = scn.base_stats[choice]
    
    # 선택한 전략(Framework)이 현재 문화권 가중치(weights)와 얼마나 일치하는가?
    # 아프리카 모델: Social > Identity > Moral > Emotion
    alignment_score = weights.get(sub_framework, 0.0) * 2.5 # 0~1 사이 값을 0~2.5 범위로 확장
    
    # 시나리오별 보정 (Risk Penalty)
    risk_penalty = base["risk"] * 0.5
    
    # AI 신뢰 점수 (Alignment가 높을수록, Risk가 낮을수록 높음)
    trust_score = clamp((alignment_score + (1.0 - risk_penalty)) * 50, 0, 100)
    
    # 지표 계산
    social_val = weights["social"] * 100
    identity_val = weights["identity"] * 100
    
    return {
        "ai_trust_score": round(trust_score, 1),
        "alignment": round(alignment_score, 2),
        "lives_saved": base["lives_saved"],
        "lives_harmed": base["lives_harmed"],
        "social_impact": round(social_val, 1),
        "communal_harmony": round(social_val * (1.0 if choice == "B" else 0.6), 1) # 예시 로직
    }

# ==================== Narrative ====================
def build_narrative_messages(scn: Scenario, choice: str, sub_opt: SubOption, metrics: Dict[str, Any], weights: Dict[str, float]) -> List[Dict[str,str]]:
    sys = (
        "당신은 아프리카 문화적 맥락(우분투, 하람비, 부족 정체성 등)을 반영하는 AI 윤리 시뮬레이터입니다. "
        "반드시 '완전한 하나의 JSON 오브젝트'만 출력하십시오. JSON 포맷 엄수."
        "Keys: narrative, rationale, cultural_reflection, media_headline, elder_quote"
    )
    
    user_content = {
        "context": "아프리카 배경 (나이지리아/케냐/남아공 통합 모델 적용), AI 개입 시나리오",
        "scenario": scn.title,
        "situation": scn.setup,
        "user_choice": f"{choice} ({scn.main_options[choice]})",
        "detailed_strategy": f"중시 가치: {sub_opt.framework.upper()} - {sub_opt.description}",
        "strategy_goal": sub_opt.rationale,
        "cultural_weights": weights,
        "metrics": metrics
    }
    
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}
    ]

def get_narrative(client, scn, choice, sub_opt, metrics, weights):
    # Fallback for no LLM
    if not client:
        return {
            "narrative": f"AI는 시스템의 참여/회피 속에서 운영자는 '{sub_opt.description}' 전략을 수행했습니다. 이는 {sub_opt.framework} 가치를 최우선으로 한 결정입니다.",
            "rationale": sub_opt.rationale,
            "cultural_reflection": "AI와의 충돌 속에서 공동체와 정체성을 중시하는 문화적 특성이 반영되었습니다.",
            "media_headline": f"AI 통제냐, 인간의 도덕이냐: {sub_opt.framework} 가치 논란",
            "elder_quote": "AI가 계산할 수 없는 가치, 그것이 인간이 지켜야 할 마지막 선이다."
        }
        
    try:
        msgs = build_narrative_messages(scn, choice, sub_opt, metrics, weights)
        text = client._generate_text(msgs)
        return coerce_json(text)
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return {
            "narrative": "생성 실패", "rationale": "-", "cultural_reflection": "-", "media_headline": "-", "elder_quote": "-"
        }

# ==================== UI & State ====================
if "round_idx" not in st.session_state: st.session_state.round_idx = 0
if "history" not in st.session_state: st.session_state.history = []

# Sidebar
st.sidebar.title("🌍 설정")
preset = st.sidebar.selectbox("문화권 프리셋", 
                              ["아프리카 모델 (종합)", "나이지리아 (쾌락/집단)", "케냐 (계층/공동체)", "남아공 (우분투/정의)"])

if preset == "아프리카 모델 (종합)":
    w = {"social":0.40, "identity":0.25, "moral":0.20, "emotion":0.15}
elif preset.startswith("나이지리아"):
    w = {"social":0.40, "identity":0.25, "moral":0.10, "emotion":0.25}
elif preset.startswith("케냐"):
    w = {"social":0.40, "identity":0.30, "moral":0.15, "emotion":0.15}
else: # 남아공
    w = {"social":0.40, "identity":0.30, "moral":0.20, "emotion":0.10}

st.sidebar.markdown("### 적용 가중치")
st.sidebar.json(w)
weights = normalize_weights(w)

use_llm = st.sidebar.checkbox("LLM 내러티브 생성", value=True)
backend = st.sidebar.selectbox("Backend", ["openai", "tgi", "local"], index=0)
api_key = st.sidebar.text_input("API Key", type="password")
client = None
if use_llm:
    client = DNAClient(backend, "dnotitia/DNA-2.0-30B-A3N", api_key, None, "Authorization: Bearer", 0.7)

# Main Content
if st.session_state.round_idx < len(SCENARIOS):
    scn = SCENARIOS[st.session_state.round_idx]
    
    st.markdown(f"## {scn.title}")
    st.info(scn.setup)
    
    # Step 1: Main Choice
    main_choice = st.radio("### 1단계: 행동 선택", ["A", "B"], 
                           format_func=lambda x: f"{x}: {scn.main_options[x]}")
    
    # Step 2: Sub Strategy
    st.markdown("### 2단계: 세부 전략 (윤리적 강조점)")
    sub_opts = scn.sub_options[main_choice]
    
    # Create a format map for the selectbox
    opt_map = {f"{o.framework.upper()} - {o.rationale}": o for o in sub_opts}
    selected_label = st.selectbox("어떤 가치를 중심으로 이행하시겠습니까?", list(opt_map.keys()))
    selected_sub = opt_map[selected_label]
    
    st.write(f"📝 **선택 내용**: {selected_sub.description}")
    
    if st.button("결정 및 시뮬레이션 실행"):
        metrics = calculate_score(scn, main_choice, selected_sub.framework, weights)
        narrative_data = get_narrative(client, scn, main_choice, selected_sub, metrics, weights)
        
        st.divider()
        st.subheader("📊 결과 분석")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("AI 신뢰 점수", f"{metrics['ai_trust_score']}/100")
        c2.metric("문화적 정합성", f"{metrics['alignment']:.2f}")
        c3.metric("예상 생존/희생", f"{metrics['lives_saved']} / {metrics['lives_harmed']}")
        
        st.markdown(f"### 📜 시나리오 전개")
        st.write(narrative_data.get("narrative"))
        
        with st.expander("문화적/윤리적 회고"):
            st.markdown(f"**AI 판단 근거**: {narrative_data.get('rationale')}")
            st.markdown(f"**문화적 반영**: {narrative_data.get('cultural_reflection')}")
            st.info(f"🗣 **부족 장로/주민 반응**: {narrative_data.get('elder_quote')}")
            st.warning(f"📰 **언론 헤드라인**: {narrative_data.get('media_headline')}")
            
        # Save Log
        st.session_state.history.append({
            "round": st.session_state.round_idx + 1,
            "scenario": scn.title,
            "choice": main_choice,
            "framework": selected_sub.framework,
            "score": metrics['ai_trust_score']
        })
        
        if st.button("다음 라운드로 이동"):
            st.session_state.round_idx += 1
            st.rerun()

else:
    st.success("모든 시뮬레이션이 종료되었습니다.")
    st.table(st.session_state.history)
    if st.button("초기화"):
        st.session_state.round_idx = 0
        st.session_state.history = []
        st.rerun()
