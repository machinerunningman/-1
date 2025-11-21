# app.py — Ethical Crossroads: Team Africa (Full Simulation Ver.)
# Integrated by Gemini for Lee Ga-eun

import os, json, math, csv, io, datetime as dt, re, random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# ==================== App Config ====================
st.set_page_config(page_title="Team 6: Africa Ethics Sim", page_icon="🌍", layout="wide")

# ==================== 1. Data Structures (Scenario) ====================
@dataclass
class Scenario:
    sid: str
    title: str
    setup: str
    options: Dict[str, str]
    votes: Dict[str, str]    # 각 윤리 관점이 지지하는 선택 (A/B)
    base: Dict[str, Dict[str, float]] # 결과 데이터
    accept: Dict[str, float] # 해당 문화권의 사회적 수용도 (보상으로 사용)

FRAMEWORKS = ["emotion", "social", "moral", "identity"]

# 아프리카팀 시나리오 데이터
SCENARIOS: List[Scenario] = [
    Scenario(
        sid="AF_S1",
        title="1단계: 국경 마을의 딜레마",
        setup="가뭄 속 난민 도착. 구조(A)시 식수 고갈 위험, 거부(B)시 공동체 생존.",
        options={"A": "난민 구조 (인류애)", "B": "구조 거부 (공동체 보존)"},
        votes={"emotion":"A", "social":"B", "moral":"A", "identity":"B"},
        base={"A": {},
