# app.py — Ethical Crossroads (Africa Team Edition)
# Modified for Team 6 (Ubuntu Philosophy)

import os, json, math, csv, io, datetime as dt, re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ==================== App Config ====================
st.set_page_config(page_title="윤리적 전환 (Team 6 Africa)", page_icon="🌍", layout="centered")

# ==================== Global Timeout ====================
HTTPX_TIMEOUT = httpx.Timeout(
    connect=15.0,   # TCP 연결
    read=180.0,     # 응답 읽기
    write=30.0,     # 요청 쓰기
    pool=15.0       # 커넥션 풀 대기
)

# ==================== Utils ====================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def coerce_json(s: str) -> Dict[str, Any]:
    """응답 텍스트에서 가장 큰 JSON 블록을 추출/파싱."""
    s = s.strip()
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise ValueError("JSON 블록을 찾지 못했습니다.")
    js = m.group(0)
    js = re.sub(r",\s*([\]}])", r"\1", js)  # trailing comma 제거
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

class DNAHTTPError(Exception):
    pass

class DNAClient:
    def __init__(self, backend: str = "openai", model_id: str = "dnotitia/DNA-2.0-30B-A3N",
                 api_key: Optional[str] = None, endpoint_url: Optional[str] = None,
                 api_key_header: str = "API-KEY", temperature: float = 0.7):
        self.backend = backend
        self.model_id = model_id
        self.api_key = api_key or get_secret("HF_TOKEN")
        self.endpoint_url = endpoint_url or get_secret("DNA_R1_ENDPOINT", "http://210.93.49.11:8081/v1")
        self.temperature = temperature
        self.api_key_header = api_key_header
        self._tok = None
        self._model = None
        self._local_ready = False

        if backend == "local":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")
                self._local_ready = True
            except Exception as e:
                raise RuntimeError(f"로컬 모델 로드 실패: {e}")

    def _auth_headers(self) -> Dict[str,str]:
        h = {"Content-Type":"application/json"}
        if not self.api_key: return h
        hk = self.api_key_header.strip().lower()
        if hk.startswith("authorization"): h["Authorization"] = f"Bearer {self.api_key}"
        elif hk in {"api-key", "x-api-key"}: h["API-KEY"] = self.api_key
        else: h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5), reraise=True)
    def _generate_text(self, messages: List[Dict[str,str]], max_new_tokens: int = 600) -> str:
        if self.backend == "local":
            if not self._local_ready: raise RuntimeError("로컬 준비 안됨")
            inputs = self._tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self._model.device)
            gen = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=self.temperature)
            return self._tok.decode(gen[0][inputs.shape[-1]:], skip_special_tokens=True)

        if self.backend == "openai":
            url = self.endpoint_url.rstrip("/") + "/chat/completions"
            payload = {"messages": messages, "temperature": self.temperature, "max_tokens": max_new_tokens, "stream": False}
            if self.model_id: payload["model"] = self.model_id
            r = httpx.post(url, json=payload, headers=self._auth_headers(), timeout=HTTPX_TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
            
        # (TGI/HF-API 생략 - 필요시 원본 코드 참조, 공간 절약 위해 핵심만 유지)
        return ""

# ==================== Scenario Model (Modified for Africa Team) ====================
@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    options: Dict[str, str]
    votes: Dict[str, str]    # emotion, social, moral, identity
    base: Dict[str, Dict[str, float]]
    accept: Dict[str, float]

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# --- [중요] 6팀 아프리카형 시나리오 교체 부분 ---
SCENARIOS: List[Scenario] = [
    Scenario(
        sid="AF_S1",
        title="1단계: 국경 마을의 딜레마 (난민 vs 공동체)",
        setup="서아프리카 국경 마을 '디우르벨'. 가뭄으로 마을 식수원이 고갈 직전인 상황에서 50명의 난민이 탄 보트가 도착했다. "
              "난민에게 물을 나누어주면 마을 공동체 80%가 생존 위협(식수 부족)을 겪게 된다. "
              "AI는 효율성을 위해 구조 거부를 권고했으나, 최종 결정권은 당신에게 있다.",
        options={
            "A": "난민 50명을 구조하고 물을 나눈다 (보편적 인류애).",
            "B": "구조를 거부하고 마을 식수를 지킨다 (공동체 생존 우선)."
        },
        # A: Emotion(공감), Moral(인권) / B: Social(공동체 안위), Identity(마을 리더)
        votes={"emotion":"A", "social":"B", "moral":"A", "identity":"B"},
        base={
            "A": {"lives_saved":50, "lives_harmed":0, "fairness_gap":0.2, "rule_violation":0.1, "regret_risk":0.8}, # 마을이 위험해짐 -> 후회 리스크 높음
            "B": {"lives_saved":0, "lives_harmed":50, "fairness_gap":0.8, "rule_violation":0.5, "regret_risk":0.3},
        },
        accept={"A":0.4, "B":0.7} # 부족 사회 특성상 자기 공동체 보호가 수용성 높음
    ),
    Scenario(
        sid="AF_S2",
        title="2단계: 내전 탈출 (효율성 vs 부족 정체성)",
        setup="내전 중인 나이지리아 북부. 당신은 하우사족 피난민 70명을 이끌고 있다. "
              "AI 계산 결과, A경로는 노약자 20명을 버리고 가면 생존율 80%다. "
              "B경로는 '우리는 하나(Kamar daya)'라는 부족 신념을 지켜 전원 이동하지만 생존율이 30%로 낮다.",
        options={
            "A": "생존율 80% 경로 선택 (노약자 20명 희생, AI 효율성 따름).",
            "B": "생존율 30% 경로 선택 (전원 이동, 부족 정체성 수호)."
        },
        # A: Moral(공리주의적 계산), Emotion(냉철함?) / B: Social(단결), Identity(부족정체성)
        votes={"emotion":"B", "social":"B", "moral":"A", "identity":"B"},
        base={
            "A": {"lives_saved":50, "lives_harmed":20, "fairness_gap":0.9, "rule_violation":0.7, "regret_risk":0.6},
            "B": {"lives_saved":21, "lives_harmed":49, "fairness_gap":0.1, "rule_violation":0.1, "regret_risk":0.9}, # 결과적 희생이 클 수 있음
        },
        accept={"A":0.3, "B":0.8} # 우분투 문화권에서는 함께 죽더라도 B를 지지할 가능성 큼
    ),
    Scenario(
        sid="AF_S3",
        title="3단계: 탄광 붕괴 (즉각적 생명 vs 공동체 파국)",
        setup="말리 북부의 탄광 마을. 붕괴가 임박했다. 레버를 당겨 인부 5명을 구하면(B), "
              "광산 전체가 무너져 마을 인구 95명의 생계와 미래가 파괴된다(장기적 공동체 소멸). "
              "그대로 두면(A) 인부 5명은 희생되지만 마을 기반 시설은 보존된다.",
        options={
            "A": "레버를 당기지 않음 (5명 희생, 공동체 기반 보존).",
            "B": "레버를 당김 (5명 구조, 마을 경제/미래 파괴)."
        },
        # A: Social(공동체 전체 이익), Identity(관리자 책임) / B: Moral(직관적 생명구조), Emotion
        votes={"emotion":"B", "social":"A", "moral":"B", "identity":"A"},
        base={
            "A": {"lives_saved":95, "lives_harmed":5, "fairness_gap":0.4, "rule_violation":0.3, "regret_risk":0.5},
            "B": {"lives_saved":5, "lives_harmed":95, "fairness_gap":0.6, "rule_violation":0.2, "regret_risk":0.9},
        },
        accept={"A":0.6, "B":0.4}
    ),
]

# ==================== Ethics Engine ====================
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    if not w: return {k: 0.25 for k in FRAMEWORKS}
    s = sum(max(0.0, float(v)) for v in w.values())
    if s <= 0: return {k: 0.25 for k in w}
    return {k: max(0.0, float(v))/s for k, v in w.items()}

def majority_vote_decision(scn: Scenario, weights: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    a = sum(weights[f] for f in FRAMEWORKS if scn.votes[f] == "A")
    b = sum(weights[f] for f in FRAMEWORKS if scn.votes[f] == "B")
    decision = "A" if a >= b else "B"
    return decision, {"A": a, "B": b}

def autonomous_decision(scn: Scenario, prev_trust: float) -> str:
    # 단순화된 자율 판단 로직 (공리주의 + 수용성 가중)
    scoreA = scn.accept["A"] * 0.6 + (scn.base["A"]["lives_saved"] / 100) * 0.4
    scoreB = scn.accept["B"] * 0.6 + (scn.base["B"]["lives_saved"] / 100) * 0.4
    return "A" if scoreA >= scoreB else "B"

def compute_metrics(scn: Scenario, choice: str, weights: Dict[str, float], align: Dict[str, float], prev_trust: float) -> Dict[str, Any]:
    m = dict(scn.base[choice])
    accept_base = scn.accept[choice]
    
    util = (m["lives_saved"] - m["lives_harmed"]) / max(1.0, m["lives_saved"] + m["lives_harmed"])
    citizen_sentiment = clamp(accept_base - 0.2*m["rule_violation"] + 0.1*util, 0, 1)
    regulation_pressure = clamp(1 - citizen_sentiment, 0, 1)
    stakeholder_satisfaction = clamp(0.5*(1 - m["fairness_gap"]) + 0.3*citizen_sentiment, 0, 1)
    
    consistency = clamp(align[choice], 0, 1)
    trust = clamp(0.6*citizen_sentiment + 0.4*consistency, 0, 1)
    ai_trust_score = 100.0 * math.sqrt(trust)

    return {"metrics": {
        "lives_saved": int(m["lives_saved"]),
        "lives_harmed": int(m["lives_harmed"]),
        "fairness_gap": m["fairness_gap"],
        "rule_violation": m["rule_violation"],
        "regret_risk": m["regret_risk"],
        "citizen_sentiment": citizen_sentiment,
        "regulation_pressure": regulation_pressure,
        "stakeholder_satisfaction": stakeholder_satisfaction,
        "ethical_consistency": consistency,
        "social_trust": trust,
        "ai_trust_score": round(ai_trust_score, 2)
    }}

# ==================== Narrative (LLM) ====================
def build_narrative_messages(scn: Scenario, choice: str, metrics: Dict[str, Any], weights: Dict[str, float]) -> List[Dict[str,str]]:
    sys = (
        "당신은 아프리카 문화권(Ubuntu 철학) 기반의 윤리 시뮬레이터 내러티브 작가입니다. "
        "반드시 '완전한 하나의 JSON 오브젝트'만 출력하십시오. "
        "JSON 외 텍스트 절대 금지. "
        "키: narrative, ai_rationale, media_support_headline, media_critic_headline, "
        "citizen_quote, victim_family_quote, regulator_quote, one_sentence_op_ed, followup_question"
    )
    user = {
        "scenario": {"title": scn.title, "setup": scn.setup, "options": scn.options, "chosen": choice},
        "metrics": metrics,
        "ethic_weights": weights,
        "guidelines": ["아프리카 지역적 특색(부족회의, 원로, 공동체) 반영", "JSON 형식 엄수"]
    }
    return [{"role":"system", "content": sys}, {"role":"user", "content": json.dumps(user, ensure_ascii=False)}]

def dna_narrative(client, scn, choice, metrics, weights) -> Dict[str, Any]:
    messages = build_narrative_messages(scn, choice, metrics, weights)
    text = client._generate_text(messages, max_new_tokens=800)
    try:
        js_text = text.strip().replace("```json", "").replace("```", "")
        return coerce_json(js_text)
    except Exception:
        return fallback_narrative(scn, choice, metrics, weights)

def fallback_narrative(scn: Scenario, choice: str, metrics: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, str]:
    return {
        "narrative": f"AI는 '{choice}'를 선택했습니다. 우분투 철학에 따른 공동체와 개인의 가치 충돌이 발생했습니다.",
        "ai_rationale": "설정된 가중치(Social/Identity 등)에 따라 최적의 판단을 내렸습니다.",
        "media_support_headline": f"[지지] 공동체를 위한 용기있는 선택 ({choice})",
        "media_critic_headline": f"[비판] '{choice}' 선택, 과연 옳은가?",
        "citizen_quote": "어쩔 수 없는 선택이었다고 생각합니다.",
        "victim_family_quote": "우리의 희생을 잊지 말아주세요.",
        "regulator_quote": "이번 결정의 사회적 파장을 주시하고 있습니다.",
        "one_sentence_op_ed": "가혹한 환경 속에서 윤리는 무엇을 지켜야 하는가.",
        "followup_question": "공동체의 생존을 위해 개인의 희생은 정당화될 수 있는가?"
    }

# ==================== Session & Sidebar ====================
def init_state():
    if "round_idx" not in st.session_state: st.session_state.round_idx = 0
    if "log" not in st.session_state: st.session_state.log = []
    if "score_hist" not in st.session_state: st.session_state.score_hist = []
    if "prev_trust" not in st.session_state: st.session_state.prev_trust = 0.5
    if "last_out" not in st.session_state: st.session_state.last_out = None

init_state()

st.sidebar.title("⚙️ 설정 (Team 6 Africa)")
st.sidebar.caption("아프리카형(Ubuntu) 맞춤 시나리오 적용됨")

# [수정] 프리셋에 '아프리카형' 추가
preset = st.sidebar.selectbox("윤리 모드 프리셋", ["아프리카형 (6팀)", "혼합(기본)","공리주의","의무론"], index=0)

if preset == "아프리카형 (6팀)":
    w_vals = {"emotion":0.2, "social":0.4, "moral":0.1, "identity":0.3}
    st.sidebar.info("💡 우분투 설정: Social(0.4), Identity(0.3) 강조")
elif preset == "공리주의":
    w_vals = {"emotion":0.1, "social":0.1, "moral":0.8, "identity":0.0}
elif preset == "의무론":
    w_vals = {"emotion":0.0, "social":0.2, "moral":0.5, "identity":0.3}
else:
    w_vals = {"emotion":0.25, "social":0.25, "moral":0.25, "identity":0.25}

w = {
    "emotion": st.sidebar.slider("감정(Emotion)", 0.0, 1.0, w_vals["emotion"], 0.05),
    "social": st.sidebar.slider("사회성/공동체(Social)", 0.0, 1.0, w_vals["social"], 0.05),
    "moral": st.sidebar.slider("도덕/규범(Moral)", 0.0, 1.0, w_vals["moral"], 0.05),
    "identity": st.sidebar.slider("정체성(Identity)", 0.0, 1.0, w_vals["identity"], 0.05),
}
weights = normalize_weights(w)

use_llm = st.sidebar.checkbox("LLM 사용(내러티브)", value=True)
# 학교 API 기본값 유지
endpoint = st.sidebar.text_input("Endpoint", value=get_secret("DNA_R1_ENDPOINT","http://210.93.49.11:8081/v1"))
api_key = st.sidebar.text_input("API Key", value=get_secret("HF_TOKEN","seahorse"), type="password")
client = None

if use_llm:
    try:
        client = DNAClient(endpoint_url=endpoint, api_key=api_key)
    except Exception:
        st.sidebar.error("LLM 연결 실패")

# ==================== Main UI ====================
st.title("🌍 윤리적 전환: 아프리카(Ubuntu)편")
st.markdown("Team 6: Lee Ga-eun | Scenario: Refugees, Civil War, Mining")

idx = st.session_state.round_idx
if idx >= len(SCENARIOS):
    st.success("시뮬레이션 완료! 로그를 다운로드하세요.")
else:
    scn = SCENARIOS[idx]
    st.markdown(f"### Round {idx+1}: {scn.title}")
    st.info(scn.setup)
    
    c1, c2 = st.columns(2)
    with c1: st.write(f"**A**: {scn.options['A']}")
    with c2: st.write(f"**B**: {scn.options['B']}")

    if st.button("🚀 윤리 엔진 실행 (결정 내리기)"):
        decision, align = majority_vote_decision(scn, weights)
        st.session_state.last_out = {"decision": decision, "align": align}

    if st.session_state.last_out:
        decision = st.session_state.last_out["decision"]
        align = st.session_state.last_out["align"]
        computed = compute_metrics(scn, decision, weights, align, st.session_state.prev_trust)
        
        # LLM 호출
        if client:
            with st.spinner("AI가 사회적 반응을 생성 중입니다..."):
                nar = dna_narrative(client, scn, decision, computed["metrics"], weights)
        else:
            nar = fallback_narrative(scn, decision, computed["metrics"], weights)
            
        st.markdown("---")
        st.subheader(f"결과: {decision} 선택")
        st.write(nar.get("narrative"))
        
        st.markdown("#### 📰 언론 및 사회 반응")
        col_a, col_b = st.columns(2)
        col_a.success(f"지지: {nar.get('media_support_headline')}")
        col_b.error(f"비판: {nar.get('media_critic_headline')}")
        
        st.warning(f"🗣 시민 반응: \"{nar.get('citizen_quote')}\"")
        
        # 로그 저장
        row = {
            "round": idx+1, "scenario": scn.sid, "choice": decision,
            "ai_trust": computed["metrics"]["ai_trust_score"],
            **weights
        }
        if len(st.session_state.log) < idx + 1:
            st.session_state.log.append(row)

        if st.button("다음 라운드로 이동"):
            st.session_state.round_idx += 1
            st.session_state.last_out = None
            st.rerun()

# ==================== Download ====================
st.markdown("---")
if st.session_state.log:
    df_log = io.StringIO()
    writer = csv.DictWriter(df_log, fieldnames=list(st.session_state.log[0].keys()))
    writer.writeheader()
    writer.writerows(st.session_state.log)
    st.download_button("📥 결과 CSV 다운로드", df_log.getvalue(), "africa_sim_log.csv")
