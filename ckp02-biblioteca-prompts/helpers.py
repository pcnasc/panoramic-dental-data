"""Funções de apoio dos notebooks: chamam o modelo local (Ollama) e guardam tudo em resultados.json."""
import json
from pathlib import Path

import ollama

MODELO = "llama3.2"  # troque pelo modelo que você tem no Ollama (ollama list)
ARQUIVO = Path(__file__).parent / "resultados.json"


def _ler():
    if ARQUIVO.exists():
        return json.loads(ARQUIVO.read_text(encoding="utf-8"))
    return {}


def _gravar(dados):
    ARQUIVO.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")


def rodar(id_prompt, etapa, prompt, system=None, temperature=0.2, max_tokens=300, modelo=MODELO):
    """Envia o prompt ao modelo, mostra a resposta e salva tudo em resultados.json.

    etapa: "v1", "v2" ou "final".
    """
    mensagens = []
    if system:
        mensagens.append({"role": "system", "content": system})
    mensagens.append({"role": "user", "content": prompt})

    resposta = ollama.chat(
        model=modelo,
        messages=mensagens,
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    texto = resposta["message"]["content"].strip()

    dados = _ler()
    dados.setdefault(id_prompt, {})[etapa] = {
        "system": system,
        "prompt": prompt,
        "output": texto,
        "params": {"modelo": modelo, "temperature": temperature, "max_tokens": max_tokens},
    }
    _gravar(dados)

    print(texto)
    return texto


def ficha(id_prompt, tecnica, objetivo, problema, risco, mitigacao):
    """Guarda os textos da ficha (diagnóstico da v1 e análise de segurança)."""
    dados = _ler()
    dados.setdefault(id_prompt, {}).update(
        {
            "tecnica": tecnica,
            "objetivo": objetivo,
            "problema": problema,
            "risco": risco,
            "mitigacao": mitigacao,
        }
    )
    _gravar(dados)
    print(f"{id_prompt} salvo em {ARQUIVO.name}")
