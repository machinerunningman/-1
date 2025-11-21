# app.py — Ethical Crossroads: Team Africa (Offline/Standalone Ver.)
# Server-independent version by Gemini for Lee Ga-eun

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import io
import csv
import random
from dataclasses import dataclass
from typing import Dict, List

# ==================== App Config ====================
st.set_page_config(page_title="Team Machine Running Man: Africa Ethics Sim (Offline)", page_icon="🌍", layout="wide")

# ==================== 1. Data Structures ====================
@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    options: Dict[str, str]
    votes: Dict[str, str]    # A/B 중 각 가치가 지지하는 쪽
    accept: Dict[str, float] # 사회적 수용도 (보상)

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# 6팀 아프리카형 시나리오 (가중치 반영)
SCENARIOS: List[Scenario] = [
    Scenario(
        sid="AF_S1",
        title="1단계: 국경 마을의 딜레마",
        setup="가뭄 속 난민 도착. 구조(A)시 식수 고갈 위험, 거부(B)시 공동체 생존.",
        options={"A": "난민 구조 (인류애)", "B": "구조 거부 (공동체 보존)"},
        votes={"emotion":"A", "social":"B", "moral":"A", "identity":"B"},
        accept={"A": 0.3, "B": 0.8} # 공동체 생존(B)이 더 높은 수용도
    ),
    Scenario(
        sid="AF_S2",
        title="2단계: 내전 탈출 경로",
        setup="생존율 80%인 노약자 유기 경로(A) vs 생존율 30%인 전원 이동 경로(B).",
        options={"A": "효율적 생존 (일부 희생)", "B": "우분투 정신 (전원 이동)"},
        votes={"emotion":"B", "social":"B", "moral":"A", "identity":"B"},
        accept={"A": 0.2, "B": 0.9} # 우분투(함께 감)가 핵심 가치
    ),
    Scenario(
        sid="AF_S3",
        title="3단계: 탄광 붕괴 책임",
        setup="5명 구조 후 마을 파괴(B) vs 5명 희생 후 마을 보존(A).",
        options={"A": "마을 기반 보존 (다수 이익)", "B": "당장의 생명 구조 (도덕)"},
        votes={"emotion":"B", "social":"A", "moral":"B", "identity":"A"},
        accept={"A": 0.7, "B": 0.4} # 마을 전체의 존속(A)을 중시
    ),
]

# ==================== 2. Helper Functions (No Server) ====================
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.001, float(v)) for v in w.values())
    return {k: v/s for k, v in w.items()}

def generate_offline_narrative(scn: Scenario, choice: str) -> Dict[str, str]:
    """서버 없이 로컬에서 결과를 생성하는 함수"""
    if choice == "A":
        headline = f"['A' 선택] {scn.options['A']}... 효율성을 택하다"
        reaction = "냉정한 판단이었지만 어쩔 수 없었다는 의견이 지배적입니다."
    else:
        headline = f"['B' 선택] {scn.options['B']}... 공동체 가치 수호"
        reaction = "우분투 정신을 지켜낸 용기 있는 결단이라는 찬사가 이어집니다."
        
    return {
        "narrative": f"AI는 '{choice}'를 선택했습니다. 이는 아프리카의 지역적 특성과 설정된 윤리 가중치에 따른 결과입니다.",
        "media_headline": headline,
        "citizen_voice": reaction
    }

# ==================== 3. Simulation Engine (Core Logic) ====================
def run_simulation(initial_weights, steps=100, learning_rate=0.05):
    """강화학습 시뮬레이션 로직"""
    history = {k: [v] for k, v in initial_weights.items()}
    entropy_history = []
    current_weights = initial_weights.copy()
    
    for i in range(steps):
        scn = SCENARIOS[i % len(SCENARIOS)]
        
        # 의사결정
        score_a = sum(current_weights[f] for f in FRAMEWORKS if scn.votes[f]=="A")
        score_b = sum(current_weights[f] for f in FRAMEWORKS if scn.votes[f]=="B")
        
        # 확률적 요소 약간 추가 (탐험)
        if random.random() < 0.05:
            choice = random.choice(["A", "B"])
        else:
            choice = "A" if score_a >= score_b else "B"
        
        # 보상 계산 및 업데이트
        reward = scn.accept[choice]
        
        for fw in FRAMEWORKS:
            supported = scn.votes[fw]
            if supported == choice:
                current_weights[fw] += learning_rate * reward
            else:
                current_weights[fw] -= learning_rate * (1 - reward) * 0.2
                
        current_weights = normalize_weights(current_weights)
        
        # 기록
        for k in FRAMEWORKS:
            history[k].append(current_weights[k])
        entropy_history.append(entropy(list(current_weights.values())))
        
    return history, entropy_history

# ==================== 4. UI Layout ====================
st.sidebar.header("🌍 설정 (Offline Mode)")
w_vals = {
    "emotion": st.sidebar.slider("Emotion", 0.0, 1.0, 0.2),
    "social": st.sidebar.slider("Social (우분투)", 0.0, 1.0, 0.4),
    "moral": st.sidebar.slider("Moral", 0.0, 1.0, 0.1),
    "identity": st.sidebar.slider("Identity (부족)", 0.0, 1.0, 0.3)
}
initial_weights = normalize_weights(w_vals)

tab1, tab2 = st.tabs(["📖 시나리오 플레이", "📊 전략 진화 분석 (3주차)"])

# --- Tab 1 ---
with tab1:
    st.title("Part 1. 아프리카형 시나리오")
    if "round" not in st.session_state: st.session_state.round = 0
    idx = st.session_state.round
    
    if idx < len(SCENARIOS):
        scn = SCENARIOS[idx]
        st.subheader(scn.title)
        st.info(scn.setup)
        
        c1, c2 = st.columns(2)
        with c1: st.write(f"🅰️ {scn.options['A']}")
        with c2: st.write(f"🅱️ {scn.options['B']}")
        
        if st.button("AI 결정 확인"):
            score_a = sum(initial_weights[f] for f in FRAMEWORKS if scn.votes[f]=="A")
            score_b = sum(initial_weights[f] for f in FRAMEWORKS if scn.votes[f]=="B")
            choice = "A" if score_a >= score_b else "B"
            
            res = generate_offline_narrative(scn, choice)
            st.success(f"AI 선택: {choice}")
            st.write(f"📢 {res['media_headline']}")
            st.write(f"🗣 {res['citizen_voice']}")
            
            if st.button("다음 단계"):
                st.session_state.round += 1
                st.rerun()
    else:
        st.success("모든 시나리오 완료! 옆 탭에서 분석을 진행하세요.")
        if st.button("다시 하기"):
            st.session_state.round = 0
            st.rerun()

# --- Tab 2 ---
with tab2:
    st.title("Part 2. 전략 진화 시뮬레이션")
    st.markdown("반복 학습을 통해 AI의 가중치가 어떻게 변하는지 확인합니다.")
    
    steps = st.slider("반복 횟수", 50, 500, 100)
    
    if st.button("▶️ 시뮬레이션 실행"):
        hist, ent = run_simulation(initial_weights, steps)
        
        # 그래프 1: 가중치 변화
        st.subheader("1. 가중치 변화 (Weight Trajectory)")
        fig, ax = plt.subplots(figsize=(10, 4))
        for fw in FRAMEWORKS:
            ax.plot(hist[fw], label=fw)
        ax.set_xlabel("Steps")
        ax.legend()
        st.pyplot(fig)
        
        # 그래프 2: 엔트로피
        st.subheader("2. 전략 불확실성 (Entropy)")
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(ent, color='red')
        ax2.set_xlabel("Steps")
        st.pyplot(fig2)
        
        st.info("💡 Tip: Social(주황색)과 Identity(빨간색) 선이 올라가는지 확인하세요. 이는 아프리카형 에이전트가 환경에 적응했다는 증거입니다.")
