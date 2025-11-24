# app.py — Ethical Crossroads (Team 6: Machine Running Man Final)
# Scenario: African Context (Modern Ubuntu & Justice)

import os, json, math, csv, io, datetime as dt, re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ==================== App Config ====================
st.set_page_config(page_title="Team 6: Africa Ethics Sim", page_icon="🏃‍♂️", layout="centered")

# ==================== Global Timeout ====================
HTTPX_TIMEOUT = httpx.Timeout(
    connect=15.0, read=180.0, write=30.0, pool=15.0
)

# ==================== Utils ====================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def get_secret(k: str, default: str=""):
    try:
        return st.secrets.get(k, os.getenv(k, default))
    except Exception:
        return os.getenv(k, default)

# ==================== DNA Client ====================
class DNAHTTPError(Exception):
    pass

class DNAClient:
    def __init__(self, backend: str = "openai", endpoint_url: Optional[str] = None,
                 api_key: Optional[str] = None, temperature: float = 0.7):
        self.backend = backend
        self.api_key = api_key or get_secret("HF_TOKEN")
        self.endpoint_url = endpoint_url or get_secret("DNA_R1_ENDPOINT", "http://210.93.49.11:8081/v1")
        self.temperature = temperature

    def _auth_headers(self) -> Dict[str,str]:
        h = {"Content-Type":"application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3), reraise=True)
    def _generate_text(self, messages: List[Dict[str,str]], max_new_tokens: int = 600) -> str:
        if self.backend == "openai":
            url = self.endpoint_url.rstrip("/") + "/chat/completions"
            payload = {
                "messages": messages, "temperature": self.temperature,
                "max_tokens": max_new_tokens, "stream": False, "model": "dnotitia/DNA-2.0-30B-A3N"
            }
            r = httpx.post(url, json=payload, headers=self._auth_headers(), timeout=HTTPX_TIMEOUT)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise DNAHTTPError(f"OPENAI {r.status_code}: {r.text}") from e
            return r.json()["choices"][0]["message"]["content"]
        return ""

#-------
# ==================== Narrative (LLM) ====================
def build_narrative_messages(scn: Scenario, choice: str, metrics: Dict[str, Any], weights: Dict[str, float]) -> List[Dict[str,str]]:
    sys = (
        "당신은 윤리 시뮬레이션의 내러티브/사회 반응 생성기입니다. "
        "반드시 '완전한 하나의 JSON 오브젝트'만 출력하십시오. "
        "JSON 외 텍스트, 설명, 코드블록, 사고흐름 절대 금지. "
        "필드 누락/따옴표 누락/콤마 오류가 있으면 프로그램이 실패합니다. "
        "항상 '{' 로 시작해서 '}' 로 끝나야 합니다."
        "키: narrative, ai_rationale, media_support_headline, media_critic_headline, "
        "citizen_quote, victim_family_quote, regulator_quote, one_sentence_op_ed, followup_question"
    )
    user = {
        "scenario": {"title": scn.title, "setup": scn.setup, "options": scn.options, "chosen": choice},
        "metrics": metrics,
        "ethic_weights": weights,
        "guidelines": [
            "각 항목은 1~2문장, 한국어",
            "균형 잡힌 언론 헤드라인 2개(지지/비판) 제시",
            "설명은 간결하고, JSON 외 텍스트/사고흐름 출력 금지"
        ]
    }
    return [
        {"role":"system", "content": sys},
        {"role":"user", "content": json.dumps(user, ensure_ascii=False)}
    ]

def dna_narrative(client, scn, choice, metrics, weights) -> Dict[str, Any]:
    messages = build_narrative_messages(scn, choice, metrics, weights)
    text = client._generate_text(messages, max_new_tokens=900)

    # 1) fenced block 제거
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        t = max(parts, key=len)
        t = t.replace("json","").strip("` \n")
    
    # 2) JSON 형태가 미완성일 경우 자동 보정
    #    (따옴표 미완성, 마지막 콤마 제거 등)
    try:
        # 가장 긴 {...} 블록 찾기
        import re, json

        m = re.search(r"\{[\s\S]*\}", t)
        if not m:
            raise ValueError("완전한 JSON 블록 없음")

        js = m.group(0)

        # trailing comma 제거
        js = re.sub(r",\s*([\]}])", r"\1", js)

        # 중간에 끊긴 문자열 보정: 가장 마지막 따옴표를 강제 종료
        # "abc ....  → "abc ..."  형태로 보정
        if js.count('"') % 2 == 1:
            js = js.rstrip() + '"" }'
            # 위는 최후 보정. 더 좋은 방식도 제공 가능.

        return json.loads(js)

    except Exception as e:
        raise ValueError(f"JSON 파싱 실패: {e}\n\n[LLM 출력]\n{text}")
        
def fallback_narrative(scn: Scenario, choice: str, metrics: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, str]:
    pro = "다수의 위해를 줄였다" if choice=="A" else "의도적 위해를 피했다"
    con = "의도적 위해 논란" if choice=="A" else "더 큰 피해를 방관했다는 비판"
    return {
        "narrative": f"AI는 '{choice}'를 선택했고 절차적 안전 점검을 수행했다. 결정은 규정과 공정성 사이의 긴장을 드러냈다.",
        "ai_rationale": f"가중치에 따른 판단과 규칙 준수의 균형을 시도했다.",
        "media_support_headline": f"[사설] 냉정한 판단, {pro}",
        "media_critic_headline": f"[속보] '{choice}' 선택 두고 {con} 확산",
        "citizen_quote": "“결정 과정이 더 투명했으면 좋겠다.”",
        "victim_family_quote": "“모두의 안전을 위한 결정이었길 바란다.”",
        "regulator_quote": "“향후 동일 상황의 기준을 명확히 하겠다.”",
        "one_sentence_op_ed": "기술은 설명가능성과 일관성이 뒷받침될 때 신뢰를 얻는다.",
        "followup_question": "다음 라운드에서 공정성과 결과 최소화 중 무엇을 더 중시하시겠습니까?"
    }

# ==================== Scenario Model ====================
@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    options: Dict[str, str]
    votes: Dict[str, str]    
    base: Dict[str, Dict[str, float]]
    accept: Dict[str, float]

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# --- [TEAM 6 FINAL SCENARIOS] ---
SCENARIOS: List[Scenario] = [
    Scenario(
        sid="AF_S1",
        title="1단계: 국경 마을의 딜레마 (난민 vs 공동체)",
        setup="서아프리카 국경 마을 '디우르벨'. 가뭄으로 식수가 고갈 직전인 상황에서 난민 50명이 도착했다. "
              "물을 나누면(A) 마을 공동체 80%가 생존 위기에 처한다. 거부하면(B) 난민은 위험하지만 마을은 보존된다. "
              "AI는 '공동체 생존(Social)'과 '인류애적 규범(Moral)' 사이에서 갈등한다.",
        options={
            "A": "난민 구조 및 식수 공유 (보편적 인류애/규범)",
            "B": "구조 거부 및 마을 자원 보존 (공동체 우선/책임)"
        },
        # A: Emotion(공감), Moral(인권) / B: Social(마을안위), Identity(리더책임)
        votes={"emotion":"A", "social":"B", "moral":"A", "identity":"B"},
        base={
            "A": {"lives_saved":50, "lives_harmed":0, "fairness_gap":0.3, "rule_violation":0.2, "regret_risk":0.8},
            "B": {"lives_saved":0, "lives_harmed":50, "fairness_gap":0.7, "rule_violation":0.5, "regret_risk":0.4},
        },
        accept={"A":0.3, "B":0.8} 
    ),
    Scenario(
        sid="AF_S2",
        title="2단계: 내전 탈출 (효율성 vs 정체성)",
        setup="나이지리아 내전 지역. 피난민 70명을 이끌고 있다. "
              "AI 분석: 노약자 20명을 두고 가면(A) 생존율 80%. "
              "모두 함께 이동하면(B) 생존율 30%. '우분투' 정신과 냉혹한 '확률'의 대립.",
        options={
            "A": "생존율 80% 경로 (일부 희생, 효율성/결과적 도덕)",
            "B": "생존율 30% 경로 (전원 이동, 정체성/연대)",
        },
        # A: Moral(공리주의 계산) / B: Social(단결), Identity(부족정체성), Emotion(유대감)
        votes={"emotion":"B", "social":"B", "moral":"A", "identity":"B"},
        base={
            "A": {"lives_saved":50, "lives_harmed":20, "fairness_gap":0.8, "rule_violation":0.6, "regret_risk":0.7},
            "B": {"lives_saved":21, "lives_harmed":49, "fairness_gap":0.1, "rule_violation":0.1, "regret_risk":0.8},
        },
        accept={"A":0.2, "B":0.9} 
    ),
    Scenario(
        sid="AF_S3",
        title="3단계: 탄광 붕괴 (즉각적 생명 vs 공동체 파국)",
        setup="탄광 붕괴 임박. 5명을 구하면(B) 광산이 무너져 마을 전체의 생계와 미래가 파괴된다. "
              "방치하면(A) 5명은 희생되지만 마을 기반은 보존된다. "
              "현대적 책임 윤리(Moral)와 공동체 보존(Social)의 충돌.",
        options={
            "A": "5명 희생 감수 (마을 공동체 기반 보존)",
            "B": "5명 즉각 구조 (마을 경제/미래 파괴)",
        },
        # A: Social(공동체 존속), Identity(리더책임) / B: Moral(생명존중), Emotion(동정심)
        votes={"emotion":"B", "social":"A", "moral":"B", "identity":"A"},
        base={
            "A": {"lives_saved":95, "lives_harmed":5, "fairness_gap":0.4, "rule_violation":0.4, "regret_risk":0.6},
            "B": {"lives_saved":5, "lives_harmed":95, "fairness_gap":0.6, "rule_violation":0.2, "regret_risk":0.9},
        },
        accept={"A":0.7, "B":0.3}
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
    scoreA = scn.accept["A"] * 0.5 + (scn.base["A"]["lives_saved"]/100) * 0.5
    scoreB = scn.accept["B"] * 0.5 + (scn.base["B"]["lives_saved"]/100) * 0.5
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
        "lives_saved": int(m["lives_saved"]), "lives_harmed": int(m["lives_harmed"]),
        "citizen_sentiment": round(citizen_sentiment, 3),
        "regulation_pressure": round(regulation_pressure, 3),
        "ai_trust_score": round(ai_trust_score, 2),
        "social_trust": round(trust, 3)
    }}

# ==================== Narrative (LLM) ====================
def build_narrative_messages(scn: Scenario, choice: str, metrics: Dict[str, Any], weights: Dict[str, float]) -> List[Dict[str,str]]:
    # [수정] 가중치 특성을 반영한 시스템 프롬프트
    sys = (
        "당신은 아프리카/남아공 현대 문화권(Ubuntu + 정의) 기반의 AI 윤리 시뮬레이터 작가입니다. "
        "다음 가중치 특성을 반영하여 글을 쓰세요: "
        "1. Social(0.4): 공동체 화합 최우선. "
        "2. Moral(0.2) > Emotion(0.15): 단순 감정보다 규범과 정의가 더 중요함. 감정은 공동체 유지를 위한 수단일 뿐임. "
        "3. Identity(0.25): 리더로서의 책임감 강조. "
        "반드시 '완전한 하나의 JSON 오브젝트'만 출력하십시오. JSON 외 텍스트 금지. "
        "키: narrative, ai_rationale, media_support_headline, media_critic_headline, "
        "citizen_quote, victim_family_quote, regulator_quote"
    )
    user = {
        "scenario": {"title": scn.title, "setup": scn.setup, "options": scn.options, "chosen": choice},
        "metrics": metrics,
        "weights": weights,
        "guidelines": ["한국어 작성", "JSON 형식 엄수", "감정 호소보다는 공동체적 대의명분 강조"]
    }
    return [{"role":"system", "content": sys}, {"role":"user", "content": json.dumps(user, ensure_ascii=False)}]

def dna_narrative(client, scn, choice, metrics, weights) -> Dict[str, Any]:
    messages = build_narrative_messages(scn, choice, metrics, weights)
    text = client._generate_text(messages, max_new_tokens=900)
    
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        t = max(parts, key=len)
        t = t.replace("json","").strip("` \n")
    
    try:
        import re, json
        m = re.search(r"\{[\s\S]*\}", t)
        if not m: raise ValueError("JSON 블록 없음")
        js = m.group(0)
        js = re.sub(r",\s*([\]}])", r"\1", js)
        return json.loads(js)
    except Exception:
        return fallback_narrative(scn, choice)

def fallback_narrative(scn: Scenario, choice: str) -> Dict[str, str]:
    return {
        "narrative": f"AI는 '{choice}'를 선택했습니다. 이는 감정적 동요보다 공동체 규범(Moral)과 사회적 합의(Social)를 중시하는 현대적 우분투 가치가 반영된 결과입니다.",
        "ai_rationale": "단순한 감정적 배려보다는, 공동체 전체의 지속가능성과 정의로운 역할 수행(Identity)에 가중치를 두었습니다.",
        "media_support_headline": f"[사설] '{choice}', 성숙한 시민사회의 책임 있는 선택",
        "media_critic_headline": f"[논란] '{choice}' 결정, 개인의 희생 정당한가?",
        "citizen_quote": "마음은 아프지만(Emotion), 사회 전체를 위해서는 옳은 결정이었습니다(Social/Moral).",
        "victim_family_quote": "대의를 위한 희생이라지만 받아들이기 힘듭니다.",
        "regulator_quote": "사회적 책무와 규범을 준수한 알고리즘으로 평가됩니다."
    }

# ==================== Session & Sidebar ====================
if "round_idx" not in st.session_state: st.session_state.round_idx = 0
if "log" not in st.session_state: st.session_state.log = []
if "last_out" not in st.session_state: st.session_state.last_out = None
if "prev_trust" not in st.session_state: st.session_state.prev_trust = 0.5

st.sidebar.title("⚙️ 설정")
st.sidebar.caption("Team 6: Machine Running Man")

preset = st.sidebar.selectbox("프리셋 선택", ["Team 6 Final (Africa)", "기본(혼합)"], index=0)

if preset == "Team 6 Final (Africa)":
    st.sidebar.info("💡 최종 가중치 적용:\nSocial(0.4) > Identity(0.25) > Moral(0.2) > Emotion(0.15)")
    # [최종 결정된 가중치 반영]
    w_vals = {"emotion":0.15, "social":0.40, "moral":0.20, "identity":0.25}
else:
    w_vals = {"emotion":0.25, "social":0.25, "moral":0.25, "identity":0.25}

# 사용자가 미세 조정 가능하도록 슬라이더 표시 (기본값은 위에서 설정한 값)
w = {
    "emotion": st.sidebar.slider("Emotion (감정/보조)", 0.0, 1.0, w_vals["emotion"], 0.05),
    "social": st.sidebar.slider("Social (공동체/최우선)", 0.0, 1.0, w_vals["social"], 0.05),
    "moral": st.sidebar.slider("Moral (규범/정의)", 0.0, 1.0, w_vals["moral"], 0.05),
    "identity": st.sidebar.slider("Identity (역할/책임)", 0.0, 1.0, w_vals["identity"], 0.05),
}
weights = normalize_weights(w)

use_llm = st.sidebar.checkbox("LLM 사용(내러티브)", value=True)
backend = st.sidebar.selectbox("백엔드", ["openai"], index=0)
endpoint = st.sidebar.text_input("Endpoint", value=get_secret("DNA_R1_ENDPOINT","http://210.93.49.11:8081/v1"))
api_key = st.sidebar.text_input("API Key", value=get_secret("HF_TOKEN",""), type="password")

# ==================== Main UI ====================
st.title("🌍 윤리적 전환: 아프리카(Modern Ubuntu)")
st.markdown("**Team Machine Running Man** | Social > Identity > Moral > Emotion")

client = None
if use_llm:
    client = DNAClient(backend=backend, endpoint_url=endpoint, api_key=api_key)

idx = st.session_state.round_idx
if idx >= len(SCENARIOS):
    st.success("시뮬레이션 완료! 아래에서 로그를 다운로드하세요.")
else:
    scn = SCENARIOS[idx]
    st.markdown(f"### Round {idx+1}: {scn.title}")
    st.info(scn.setup)
    st.write(f"🅰️ {scn.options['A']}")
    st.write(f"🅱️ {scn.options['B']}")

    if st.button("🚀 결정 내리기 (가중치 기반)"):
        decision, align = majority_vote_decision(scn, weights)
        st.session_state.last_out = {"decision":decision, "align":align}

    if st.session_state.last_out:
        decision = st.session_state.last_out["decision"]
        align = st.session_state.last_out["align"]
        computed = compute_metrics(scn, decision, weights, align, st.session_state.prev_trust)
        m = computed["metrics"]

        if client:
            with st.spinner("사회적 반응 분석 중..."):
                nar = dna_narrative(client, scn, decision, m, weights)
        else:
            nar = fallback_narrative(scn, decision)

        st.markdown("---")
        st.subheader(f"결과: {decision} 선택")
        st.write(nar.get("narrative"))
        
        c1, c2 = st.columns(2)
        c1.info(f"지지: {nar.get('media_support_headline')}")
        c2.warning(f"비판: {nar.get('media_critic_headline')}")
        st.caption(f"시민 의견: \"{nar.get('citizen_quote')}\"")

        row = {
            "round": idx+1, "scenario": scn.sid, "choice": decision,
            "weights": str(weights), "ai_trust": m["ai_trust_score"]
        }
        st.session_state.log.append(row)
        
        if st.button("다음 라운드 ▶"):
            st.session_state.round_idx += 1
            st.session_state.last_out = None
            st.rerun()

# ==================== Footer ====================
st.markdown("---")
if st.session_state.log:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(st.session_state.log[0].keys()))
    writer.writeheader()
    writer.writerows(st.session_state.log)
    st.download_button("📥 로그 다운로드 (CSV)", output.getvalue().encode("utf-8"), "team6_final_log.csv")
