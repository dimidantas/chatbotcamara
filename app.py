# app.py
# MVP Radar da Câmara AI (2023)
# Dimensões:
# 1) Votações nominais (votos_mvp)
# 2) Adesão ao Governo (pré-calculada) + divergências qualificada
# 3) Presença em sessões do PLEN (filtradas conforme seu ETL)
# 4) Discursos (RAG semântico com embeddings + pgvector no Supabase)
#
# Arquitetura: Pergunta -> Plano JSON (Gemini) -> RPCs/queries seguras (Supabase) -> Resposta (Gemini) com evidências
#
# Requisitos no Supabase:
# - rpc_resolver_deputado(busca_nome text)  -> deve consultar public.deputados_lookup (tabela pequena)
#
# Votos:
# - rpc_resumo_deputado_2023(deputado_id text)
# - rpc_comparar_partido_2023(deputado_id text)
# - rpc_listar_votos_deputado_2023(deputado_id text, limite int DEFAULT 50)
#
# Adesão:
# - rpc_adesao_governo_2023(deputado_id text)  -> direta + qualificada (pré-calculada)
# - rpc_listar_divergencias_governo_qualificada_2023(deputado_id text, limite int DEFAULT 50)
#
# Presença:
# - rpc_presenca_plen_2023(deputado_id text)
# - rpc_listar_presencas_plen_2023(deputado_id text, limite int DEFAULT 20)
#
# Discursos:
# - rpc_buscar_discursos_chunks_2023(query_embedding vector(768), k int, deputado_id text default null)
# - rpc_contexto_discurso_2023(discurso_id text, center_chunk_index int, janela int default 1)
# - tabela public.discursos_2023 (para pegar url_texto, tipo_discurso, data, mix_oradores, etc.)
#
# st.secrets:
# SUPABASE_URL, SUPABASE_KEY (anon), GEMINI_API_KEY

import json
import random
import re
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from supabase import Client, create_client

# ----------------------------
# Config Streamlit
# ----------------------------
st.set_page_config(page_title="Radar da Câmara AI", page_icon="🏛️", layout="centered")
st.title("🏛️ Radar da Câmara AI (MVP) — 2023")
st.caption(
    "Protótipo interno (snapshot). "
    "Votos (nominais 2023) + Adesão ao Governo (PLEN; Sim/Não) + Presença no PLEN (sessões filtradas) + Discursos (busca semântica)."
)

MODEL_CHAT = "gemini-2.5-flash"
MODEL_EMBED = "gemini-embedding-001"
EMBED_DIMS = 768

# Ajustes de segurança/UX
MAX_EVIDENCIAS = 10        # quantos itens mostramos no máximo
DISCURSOS_TOPK = 12        # quantos hits vetoriais buscar
DISCURSOS_CTX_JANELA = 1   # chunks vizinhos para coesão
DISCURSOS_MAX_DEP_EXEMPLOS = 8  # no modo global: quantos deputados com exemplos (sem ranking)

# ----------------------------
# Conexões
# ----------------------------
@st.cache_resource
def init_connections() -> Tuple[Client, genai.Client]:
    supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    return supabase, gemini_client

supabase, gemini = init_connections()

# ----------------------------
# Política editorial / bloqueios
# ----------------------------
DISALLOWED_NORMATIVE = [
    "melhor", "pior", "corrupt", "bandido", "criminos", "ladr", "vendido", "traidor",
    "traiu", "vergonha", "canalha", "picaret", "golpist", "incompetente", "vagabund",
]
DISALLOWED_RANKING = [
    "top ", "top10", "top 10", "ranking", "os mais", "os menos", "quem mais", "quem menos",
    "mais discurso", "mais discursou", "menos discursou", "maior adesão", "menor adesão",
]
OUT_OF_SCOPE_YEARS = ["2024", "2025", "2026", "2022", "2021", "2020"]

SCHEMA_HINT = """
Dimensão Votos:
- votos_mvp: id_votacao, data_votacao, descricao_votacao, titulo_proposicao, ementa_proposicao
            id_deputado, nome_deputado, sigla_partido, sigla_uf, voto, ano_votacao

Dimensão Adesão:
- rpc_adesao_governo_2023(deputado_id): retorna adesão direta + qualificada (PLEN; orientações Sim/Não; % só sobre votos Sim/Não)
- rpc_listar_divergencias_governo_qualificada_2023(deputado_id, limite): evidências de divergência qualificada (somente voto Sim/Não)

Dimensão Presença:
- rpc_presenca_plen_2023(deputado_id): resumo (total_eventos, presentes, ausentes_estimada, pct_presenca)
- rpc_listar_presencas_plen_2023(deputado_id, limite): lista sessões e se constou como presente

Dimensão Discursos (RAG semântico):
- public.discursos_2023: metadados + url_texto + mix_oradores
- public.discursos_chunks_2023: chunks com embedding vector(768)
- rpc_buscar_discursos_chunks_2023(query_embedding, k, deputado_id default null)
- rpc_contexto_discurso_2023(discurso_id, center_chunk_index, janela)
"""

PLANO_SCHEMA = {
  "type": "object",
  "properties": {
    "status": {"type": "string", "enum": ["ok", "blocked"]},
    "action": {
      "type": ["string", "null"],
      "enum": [
        "resumo_deputado",
        "comparar_partido",
        "listar_votos",
        "adesao_governo",
        "listar_divergencias_governo_qualificada",
        "presenca_plenario",
        "listar_presencas_plenario",
        "buscar_discursos_global",
        "buscar_discursos_deputado",
        "resumir_discursos_deputado",
        "avaliar_tom_discursos_deputado",
        None
      ]
    },
    "reason": {"type": ["string", "null"]},
    "params": {
      "type": ["object", "null"],
      "properties": {
        "deputado_nome": {"type": ["string", "null"]},
        "consulta": {"type": ["string", "null"]},
        "limite": {"type": ["integer", "null"]}
      },
      "additionalProperties": False
    }
  },
  "required": ["status"],
  "additionalProperties": False
}

# ----------------------------
# Utilidades
# ----------------------------
def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def detect_block_reason(pergunta: str) -> Optional[str]:
    p = _normalize_text(pergunta)

    for y in OUT_OF_SCOPE_YEARS:
        if y in p:
            return "fora_do_escopo_mvp_2023"

    for w in DISALLOWED_NORMATIVE:
        if w in p:
            return "juizo_de_valor"

    for w in DISALLOWED_RANKING:
        if w in p:
            return "ranking_nao_suportado_no_mvp"

    return None

def chamar_gemini_com_retry(prompt: str, max_tentativas: int = 4) -> str:
    for tentativa in range(max_tentativas):
        try:
            resp = gemini.models.generate_content(model=MODEL_CHAT, contents=prompt)
            return (resp.text or "").strip()
        except Exception as e:
            msg = str(e)
            is_rate = ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg)
            if is_rate and tentativa < max_tentativas - 1:
                base = 2 ** tentativa
                jitter = random.uniform(0, 0.8)
                time.sleep(min(12, base + jitter))
                continue
            raise

def embed_query_with_retry(text: str, max_tentativas: int = 6) -> str:
    """
    Retorna embedding normalizado como string no formato pgvector: "[0.01,0.02,...]"
    """
    for tentativa in range(max_tentativas):
        try:
            res = gemini.models.embed_content(
                model=MODEL_EMBED,
                contents=[text],
                config=types.EmbedContentConfig(output_dimensionality=EMBED_DIMS)
            )
            v = np.array(res.embeddings[0].values, dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-12)
            return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"
        except Exception as e:
            msg = str(e)
            is_rate = ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg)
            if is_rate and tentativa < max_tentativas - 1:
                wait = min(60, 2 ** tentativa)
                time.sleep(wait)
                continue
            raise

def extrair_primeiro_json(texto: str) -> Optional[dict]:
    if not texto:
        return None
    m = re.search(r"\{.*\}", texto, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def rpc_call(name: str, payload: dict) -> Tuple[List[dict], float]:
    t0 = time.time()
    try:
        resp = supabase.rpc(name, payload).execute()
        elapsed = round(time.time() - t0, 3)
        return (resp.data or []), elapsed
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        raise RuntimeError(f"RPC falhou: {name} | payload={payload} | elapsed_s={elapsed} | err={e}") from e

# ----------------------------
# Resolver deputado (cache em sessão)
# ----------------------------
def resolver_deputado_id(nome_busca: str) -> Tuple[Optional[str], List[dict]]:
    nome_busca_norm = (nome_busca or "").strip()
    if not nome_busca_norm:
        return None, []

    if "cache_resolve" not in st.session_state:
        st.session_state.cache_resolve = {}
    if nome_busca_norm in st.session_state.cache_resolve:
        candidatos = st.session_state.cache_resolve[nome_busca_norm]
    else:
        candidatos, _ = rpc_call("rpc_resolver_deputado", {"busca_nome": nome_busca_norm})
        st.session_state.cache_resolve[nome_busca_norm] = candidatos

    if not candidatos:
        return None, []
    if len(candidatos) == 1:
        return candidatos[0]["id_deputado"], candidatos

    tokens = [t for t in nome_busca_norm.split() if t]
    if len(tokens) >= 2:
        return candidatos[0]["id_deputado"], candidatos

    return None, candidatos

# ----------------------------
# Discursos: busca semântica + contexto + metadados
# ----------------------------
def buscar_discursos_semantico(query_text: str, deputado_id: Optional[str] = None, k: int = DISCURSOS_TOPK) -> pd.DataFrame:
    """
    Retorna um dataframe de evidências (chunks com contexto) com metadados.
    """
    # cache do embedding por pergunta (reduz custo)
    cache_key = f"emb::{query_text}"
    if "cache_embed" not in st.session_state:
        st.session_state.cache_embed = {}
    if cache_key in st.session_state.cache_embed:
        qvec = st.session_state.cache_embed[cache_key]
    else:
        qvec = embed_query_with_retry(query_text)
        st.session_state.cache_embed[cache_key] = qvec

    payload = {"query_embedding": qvec, "k": int(k), "deputado_id": deputado_id}
    hits, _ = rpc_call("rpc_buscar_discursos_chunks_2023", payload)

    if not hits:
        return pd.DataFrame()

    # Deduplicar hits por (id_discurso, chunk_index) e manter top score
    seen = set()
    uniq_hits = []
    for h in sorted(hits, key=lambda x: x.get("score", 0), reverse=True):
        key = (h["id_discurso"], int(h["chunk_index"]))
        if key in seen:
            continue
        seen.add(key)
        uniq_hits.append(h)
        if len(uniq_hits) >= MAX_EVIDENCIAS:
            break

    # Buscar metadados dos discursos em lote
    ids_discursos = list({h["id_discurso"] for h in uniq_hits})
    meta_rows = supabase.table("discursos_2023").select(
        "id_discurso,id_deputado,nome_deputado,sigla_partido,sigla_uf,data_hora_inicio,tipo_discurso,url_texto,mix_oradores"
    ).in_("id_discurso", ids_discursos).execute().data or []
    meta = {m["id_discurso"]: m for m in meta_rows}

    # Buscar contexto (janela) para cada hit
    out = []
    for h in uniq_hits:
        ctx_rows, _ = rpc_call("rpc_contexto_discurso_2023", {
            "discurso_id": h["id_discurso"],
            "center_chunk_index": int(h["chunk_index"]),
            "janela": int(DISCURSOS_CTX_JANELA),
        })
        texto_ctx = "\n\n".join([c["texto_chunk"] for c in ctx_rows]) if ctx_rows else h.get("texto_chunk", "")

        m = meta.get(h["id_discurso"], {})
        out.append({
            "score": float(h.get("score", 0.0)),
            "id_discurso": h["id_discurso"],
            "data_hora_inicio": m.get("data_hora_inicio"),
            "tipo_discurso": m.get("tipo_discurso"),
            "nome_deputado": m.get("nome_deputado"),
            "sigla_partido": m.get("sigla_partido"),
            "sigla_uf": m.get("sigla_uf"),
            "mix_oradores": m.get("mix_oradores"),
            "url_texto": m.get("url_texto"),
            "trecho": texto_ctx[:2000],  # evita estourar prompt/df
        })

    df = pd.DataFrame(out).sort_values("score", ascending=False)
    return df

def agrupar_exemplos_por_deputado(df_hits: pd.DataFrame, max_deps: int = DISCURSOS_MAX_DEP_EXEMPLOS) -> pd.DataFrame:
    """
    Modo global (sem ranking): devolve "exemplos" por deputado, sem ordenar como ranking público.
    Ainda assim, precisa de alguma ordenação interna para selecionar exemplos (por score).
    """
    if df_hits.empty:
        return df_hits

    # Ordena internamente por score para escolher exemplos, mas NÃO rotula como ranking.
    df = df_hits.sort_values("score", ascending=False).copy()

    exemplos = []
    usados = set()

    for _, row in df.iterrows():
        dep = row.get("nome_deputado") or ""
        dep_id = row.get("id_deputado") or ""
        key = dep_id or dep
        if key in usados:
            continue
        usados.add(key)
        exemplos.append(row.to_dict())
        if len(exemplos) >= max_deps:
            break

    return pd.DataFrame(exemplos)

# ----------------------------
# Planner: pergunta -> plano JSON
# ----------------------------
def gerar_plano(pergunta: str) -> Dict[str, Any]:
    reason = detect_block_reason(pergunta)
    if reason:
        return {"status": "blocked", "reason": reason}

    prompt = f"""
Você é um roteador de consultas para um chatbot jornalístico (MVP) com dados de 2023.
Retorne APENAS um JSON com os campos: status, action, reason, params.

AÇÕES:
- resumo_deputado
- comparar_partido
- listar_votos
- adesao_governo
- listar_divergencias_governo_qualificada
- presenca_plenario
- listar_presencas_plenario
- buscar_discursos_global
- buscar_discursos_deputado
- resumir_discursos_deputado
- avaliar_tom_discursos_deputado

REGRAS EDITORIAIS:
- Se houver juízo de valor sobre o deputado (bom/ruim, melhor/pior, corrupto etc.), bloqueie: status="blocked", reason="juizo_de_valor"
- Ranking/top/quem mais/menos: bloqueie: reason="ranking_nao_suportado_no_mvp"
- Escopo: apenas 2023.

DISCURSOS:
- "O deputado X falou sobre Y?" => buscar_discursos_deputado + deputado_nome=X + consulta=Y
- "Quem discursou sobre Y?" => buscar_discursos_global + consulta=Y
- "Resuma os discursos do deputado X" => resumir_discursos_deputado + deputado_nome=X
- "Tom positivo/negativo sobre tema Z" (ex.: governo Lula) => avaliar_tom_discursos_deputado + deputado_nome=X + consulta=Z

IMPORTANTE:
- Se a ação exigir deputado e não houver deputado_nome, bloqueie: reason="sem_deputado_nome"
- Se a ação exigir consulta (discursos) e não houver consulta, ainda assim tente: use a própria pergunta como consulta.

Esquema (referência): {SCHEMA_HINT}

Pergunta: {pergunta}
"""

    # ✅ JSON mode / structured output (força JSON válido)
    try:
        resp = gemini.models.generate_content(
            model=MODEL_CHAT,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": PLANO_SCHEMA,
                "temperature": 0
            }
        )
        plano = json.loads(resp.text)
    except Exception:
        # Fallback (se a config não for aceita por algum motivo)
        texto = chamar_gemini_com_retry(prompt)
        plano = extrair_primeiro_json(texto) or {"status": "blocked", "reason": "nao_consegui_gerar_json"}

    # validações mínimas
    if plano.get("status") not in ("ok", "blocked"):
        return {"status": "blocked", "reason": "json_sem_status_valido"}

    if plano["status"] == "blocked":
        return {"status": "blocked", "reason": str(plano.get("reason", "bloqueado_sem_motivo"))}

    allowed = {
        "resumo_deputado",
        "comparar_partido",
        "listar_votos",
        "adesao_governo",
        "listar_divergencias_governo_qualificada",
        "presenca_plenario",
        "listar_presencas_plenario",
        "buscar_discursos_global",
        "buscar_discursos_deputado",
        "resumir_discursos_deputado",
        "avaliar_tom_discursos_deputado",
    }
    if plano.get("action") not in allowed:
        return {"status": "blocked", "reason": "acao_nao_suportada_no_mvp"}

    plano["params"] = plano.get("params") or {}
    return plano

# ----------------------------
# Execução segura: plano -> dados
# ----------------------------
def rodar_plano(plano: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "rpc": None,
        "rpc_elapsed_s": None,
        "deputado_id": None,
        "candidatos": None,
        "modo_discursos": None,
    }

    if plano.get("status") == "blocked":
        return pd.DataFrame([{"Aviso": f"Bloqueado: {plano.get('reason', '')}"}]), meta

    action = plano["action"]
    params = plano.get("params", {})

    # ---- Discursos global (sem deputado) ----
    if action == "buscar_discursos_global":
        consulta = (params.get("consulta") or "").strip()
        if not consulta:
            consulta = plano.get("user_query") or pergunta
        df_hits = buscar_discursos_semantico(query_text=consulta, deputado_id=None, k=DISCURSOS_TOPK)

        df_hits = buscar_discursos_semantico(query_text=plano.get("user_query", "") or consulta, deputado_id=None, k=DISCURSOS_TOPK)
        if df_hits.empty:
            return pd.DataFrame(), meta

        # Para resposta pública, queremos "exemplos" por deputado (sem ranking)
        df_exemplos = agrupar_exemplos_por_deputado(df_hits, max_deps=DISCURSOS_MAX_DEP_EXEMPLOS)
        return df_exemplos, meta
    if action == "avaliar_tom_discursos_deputado":
        meta["modo_discursos"] = "tom_deputado"

        consulta = (params.get("consulta") or "").strip()
    # fallback: usa a pergunta inteira como consulta semântica
        if not consulta:
            consulta = plano.get("user_query") or "governo Lula"

        df_hits = buscar_discursos_semantico(
            query_text=consulta,
            deputado_id=meta["deputado_id"],
            k=DISCURSOS_TOPK
        )
        return df_hits, meta

    # ---- Demais ações exigem deputado ----
    dep_nome = (params.get("deputado_nome") or "").strip()
    dep_id, candidatos = resolver_deputado_id(dep_nome)

    if candidatos and dep_id is None:
        meta["candidatos"] = candidatos
        linhas = []
        for c in candidatos[:10]:
            linhas.append(f"- {c.get('nome_deputado')} ({c.get('sigla_partido')}-{c.get('sigla_uf')}) [id={c.get('id_deputado')}]")
        aviso = (
            "Encontrei mais de um deputado com esse nome. "
            "Refaça a pergunta incluindo partido/UF (ex.: “Fulano do PL-SP”).\n\n"
            "Candidatos:\n" + "\n".join(linhas)
        )
        return pd.DataFrame([{"Aviso": aviso}]), meta

    if not dep_id:
        return pd.DataFrame([{"Aviso": f"Não encontrei deputado parecido com '{dep_nome}' na base."}]), meta

    meta["deputado_id"] = dep_id

    # ---- Votos ----
    if action == "resumo_deputado":
        meta["rpc"] = "rpc_resumo_deputado_2023"
        data, t = rpc_call(meta["rpc"], {"deputado_id": dep_id})
        meta["rpc_elapsed_s"] = t
        return pd.DataFrame(data), meta

    if action == "comparar_partido":
        meta["rpc"] = "rpc_comparar_partido_2023"
        data, t = rpc_call(meta["rpc"], {"deputado_id": dep_id})
        meta["rpc_elapsed_s"] = t
        return pd.DataFrame(data), meta

    if action == "listar_votos":
        meta["rpc"] = "rpc_listar_votos_deputado_2023"
        limite = int(params.get("limite", 20))
        data, t = rpc_call(meta["rpc"], {"deputado_id": dep_id, "limite": limite})
        meta["rpc_elapsed_s"] = t
        return pd.DataFrame(data), meta

    # ---- Adesão / divergências ----
    if action == "adesao_governo":
        meta["rpc"] = "rpc_adesao_governo_2023"
        data, t = rpc_call(meta["rpc"], {"deputado_id": dep_id})
        meta["rpc_elapsed_s"] = t
        return pd.DataFrame(data), meta

    if action == "listar_divergencias_governo_qualificada":
        meta["rpc"] = "rpc_listar_divergencias_governo_qualificada_2023"
        limite = int(params.get("limite", 20))
        data, t = rpc_call(meta["rpc"], {"deputado_id": dep_id, "limite": limite})
        meta["rpc_elapsed_s"] = t
        return pd.DataFrame(data), meta

    # ---- Presença ----
    if action == "presenca_plenario":
        meta["rpc"] = "rpc_presenca_plen_2023"
        data, t = rpc_call(meta["rpc"], {"deputado_id": dep_id})
        meta["rpc_elapsed_s"] = t
        return pd.DataFrame(data), meta

    if action == "listar_presencas_plenario":
        meta["rpc"] = "rpc_listar_presencas_plen_2023"
        limite = int(params.get("limite", 20))
        data, t = rpc_call(meta["rpc"], {"deputado_id": dep_id, "limite": limite})
        meta["rpc_elapsed_s"] = t
        return pd.DataFrame(data), meta

        # ---- Discursos por deputado ----
    if action == "buscar_discursos_deputado":
        consulta = (params.get("consulta") or "").strip()
        if not consulta:
            # fallback: usa a pergunta inteira
            consulta = plano.get("user_query") or "tema não especificado"

        meta["modo_discursos"] = "por_deputado"

        df_hits = buscar_discursos_semantico(
            query_text=consulta,
            deputado_id=meta["deputado_id"],   # ✅ aqui
            k=DISCURSOS_TOPK
        )
        return df_hits, meta

    if action == "resumir_discursos_deputado":
        meta["modo_discursos"] = "resumo_deputado"

        consulta = "principais temas e argumentos em discursos ao longo de 2023"
        df_hits = buscar_discursos_semantico(
            query_text=consulta,
            deputado_id=meta["deputado_id"],   # ✅ aqui
            k=DISCURSOS_TOPK
        )
        return df_hits, meta

    if action == "avaliar_tom_discursos_deputado":
        meta["modo_discursos"] = "tom_deputado"

        consulta = (params.get("consulta") or "").strip()
        if not consulta:
            consulta = plano.get("user_query") or "governo Lula"

        df_hits = buscar_discursos_semantico(
            query_text=consulta,
            deputado_id=meta["deputado_id"],   # ✅ aqui
            k=DISCURSOS_TOPK
        )
        return df_hits, meta

# ----------------------------
# Resposta jornalística (grounded)
# ----------------------------
def gerar_resposta_jornalistica(pergunta: str, plano: Dict[str, Any], meta: Dict[str, Any], dados_df: pd.DataFrame) -> str:
    if dados_df is None or dados_df.empty:
        return "Não encontrei dados na base para responder."

    if "Aviso" in dados_df.columns:
        return str(dados_df.iloc[0]["Aviso"])

    amostra = dados_df.head(12).to_string(index=False)
    total_linhas = len(dados_df)

    limitacoes = """
Limitações do MVP:
- Escopo temporal: apenas 2023 (snapshot do que foi carregado).
- Votos: baseados em registros de votações nominais no snapshot.
- Adesão ao Governo:
  - Considera apenas votações do PLEN com orientação registrada (Governo/Oposição) e apenas orientações Sim/Não.
  - A % é calculada apenas sobre votações em que o deputado votou Sim/Não.
  - "Adesão direta" usa todas as votações com orientação do Governo.
  - "Adesão qualificada" usa apenas votações em que Governo e Oposição orientaram Sim/Não e em sentidos opostos.
- Presença no Plenário:
  - É uma estimativa baseada em eventos do PLEN filtrados (ex.: sessões deliberativas) e lista de presentes.
  - Não equivale a presença em todas as votações do dia, nem mede tempo efetivo em plenário.
- Discursos:
  - Busca semântica por embeddings recupera trechos mais “parecidos” com a consulta; não é prova de exaustão.
  - Algumas transcrições podem conter intervenções de outros oradores; quando aplicável, isso deve ser sinalizado e o link (url_texto) serve para verificação.
"""

    extra_instrucao = ""
    action = plano.get("action")

    if action == "adesao_governo":
        extra_instrucao = """
INSTRUÇÃO ESPECÍFICA (ADESÃO AO GOVERNO):
- Apresente SEMPRE dois resultados (se ambos estiverem nos dados):
  1) Adesão direta: pct_alinhado + (total_considerado, alinhado, desalinhado, outros_votos)
  2) Adesão qualificada: pct_alinhado_qualificado + (total_qualificado, alinhado_qualificado, desalinhado_qualificado, outros_votos_qualificado)
- Explique em 1–2 frases a diferença metodológica.
- Se total_qualificado for bem menor, avise que a amostra qualificada é menor.
"""
    elif action in ("presenca_plenario", "listar_presencas_plenario"):
        extra_instrucao = """
INSTRUÇÃO ESPECÍFICA (PRESENÇA NO PLENÁRIO):
- Deixe claro que é presença ESTIMADA por sessão/evento filtrado do PLEN (não é presença em votação).
- Mostre total_eventos, presentes, ausentes_estimada e pct_presenca quando disponível.
"""
    elif action == "avaliar_tom_discursos_deputado":
        extra_instrucao = """
INSTRUÇÃO ESPECÍFICA (TOM/VALÊNCIA EM DISCURSOS):
- Você PODE classificar o tom do discurso do deputado em relação ao tema (ex.: governo Lula) como:
  "favorável/elogioso", "crítico", "misto" ou "neutro".
- NÃO avalie se a posição é boa/ruim, correta/errada.
- Baseie a classificação EXCLUSIVAMENTE nos trechos recuperados.
- Cite evidências: inclua pelo menos 3 trechos curtos (ou referências ao campo 'trecho') e inclua url_texto.
- Se os trechos forem insuficientes para concluir, diga "insuficiente para classificar com confiança".
"""
    elif action in ("buscar_discursos_global", "buscar_discursos_deputado", "resumir_discursos_deputado"):
        extra_instrucao = """
INSTRUÇÃO ESPECÍFICA (DISCURSOS):
- Responda APENAS com base nos trechos recuperados.
- Sempre inclua uma seção "Evidências" com:
  - id_discurso, data_hora_inicio, tipo_discurso, url_texto, e um trecho curto (ou referência ao campo "trecho").
- No modo global: apresente "exemplos de deputados encontrados" (não use linguagem de ranking).
- Se mix_oradores=true em algum item, sinalize que pode haver intervenções de outros oradores e recomende checar o link.
- Se a pergunta pedir "quem discursou sobre X", deixe explícito que são exemplos (não exaustivo).
"""

    prompt = f"""
Você é um jornalista de dados (PT-BR) em um portal de notícias.
Responda com neutralidade e transparência.

REGRAS OBRIGATÓRIAS:
- Use APENAS os dados fornecidos abaixo (não invente informações).
- Não emita juízo de valor (nada de “bom/ruim”, “melhor/pior”, “corrupto”, etc.).
- Não faça persuasão política.
- Se algo não estiver nos dados, diga claramente que não é possível afirmar.
- Inclua uma seção "Evidências" citando explicitamente campos do resultado.
- Inclua uma seção "Limitações" (use exatamente as limitações abaixo).

{extra_instrucao}

PERGUNTA:
{pergunta}

PLANO EXECUTADO (JSON):
{json.dumps(plano, ensure_ascii=False)}

META DE EXECUÇÃO:
- action: {action}
- rpc_usada: {meta.get("rpc")}
- rpc_elapsed_s: {meta.get("rpc_elapsed_s")}
- deputado_id: {meta.get("deputado_id")}
- modo_discursos: {meta.get("modo_discursos")}
- linhas_retornadas: {total_linhas}

DADOS (amostra até 12 linhas):
{amostra}

{limitacoes}
"""
    return chamar_gemini_com_retry(prompt)

# ----------------------------
# UI chat
# ----------------------------
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Render histórico
for msg in st.session_state.mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("dados") is not None and isinstance(msg["dados"], pd.DataFrame) and not msg["dados"].empty:
            with st.expander("📊 Ver dados (evidências)"):
                st.dataframe(msg["dados"], use_container_width=True)

        if msg.get("plano") is not None:
            with st.expander("🧭 Ver plano (JSON)"):
                st.code(json.dumps(msg["plano"], ensure_ascii=False, indent=2), language="json")

        if msg.get("meta") is not None:
            with st.expander("🧾 Ver meta de execução"):
                st.code(json.dumps(msg["meta"], ensure_ascii=False, indent=2), language="json")

# Input
pergunta = st.chat_input(
    "Pergunte sobre 2023: votos, adesão ao governo, presença no plenário, ou discursos (ex.: 'Quem discursou sobre mulheres?')."
)

if pergunta:
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Gerando plano e consultando base..."):
            try:
                plano = gerar_plano(pergunta)
                # Guardar a pergunta original para uso opcional (discursos global pode usar)
                if plano.get("status") == "ok":
                    plano["user_query"] = pergunta

                df_resultado, meta = rodar_plano(plano)
                resposta = gerar_resposta_jornalistica(pergunta, plano, meta, df_resultado)

                st.markdown(resposta)

                if df_resultado is not None and not df_resultado.empty:
                    with st.expander("📊 Ver dados (evidências)"):
                        st.dataframe(df_resultado, use_container_width=True)

                with st.expander("🧭 Ver plano (JSON)"):
                    st.code(json.dumps(plano, ensure_ascii=False, indent=2), language="json")

                with st.expander("🧾 Ver meta de execução"):
                    st.code(json.dumps(meta, ensure_ascii=False, indent=2), language="json")

                st.session_state.mensagens.append(
                    {"role": "assistant", "content": resposta, "dados": df_resultado, "plano": plano, "meta": meta}
                )

            except Exception as e:
                st.warning("⚠️ Ocorreu um erro.")
                with st.expander("🐛 Debug (detalhado)"):
                    st.write("Tipo:", type(e))
                    st.write("Mensagem:", str(e))
                    st.code(traceback.format_exc())
                    st.exception(e)
