"""
Pipeline de EDA — Análise Exploratória de Dados.
Versão 1: informações básicas.

Uso:
    python eda.py
"""

import os
import pandas as pd


# -----------------------------------------------------------------------------
# CARREGAMENTO
# -----------------------------------------------------------------------------

def carregar_dataset(caminho_csv: str) -> pd.DataFrame:
    return pd.read_csv(caminho_csv)


# -----------------------------------------------------------------------------
# ANÁLISES
# -----------------------------------------------------------------------------

def verificar_id(df: pd.DataFrame) -> str | None:
    """Verifica se a primeira coluna é um ID.

    Checa se a sílaba 'id' aparece no nome da primeira coluna.
    Retorna o nome da coluna se for ID, ou None se não for.
    """
    primeira_coluna = df.columns[0]
    if "id" in primeira_coluna.lower():
        return primeira_coluna
    return None


def info_basicas(df: pd.DataFrame, coluna_id: str | None) -> str:
    linhas, colunas = df.shape

    # Monta a linha sobre ID dependendo do resultado de verificar_id
    if coluna_id:
        linha_id = f"Coluna de ID detectada: {coluna_id}"
    else:
        linha_id = "Nenhuma coluna de ID detectada."

    linhas_texto = [
        "=== INFORMAÇÕES BÁSICAS ===",
        linha_id,
        f"Linhas    : {linhas}",
        f"Colunas   : {colunas}",
        "",
        "--- Colunas e seus tipos ---",
    ]

    for nome, tipo in df.dtypes.items():
        linhas_texto.append(f"  {nome:<35} {str(tipo)}")

    return "\n".join(linhas_texto)


# -----------------------------------------------------------------------------
# SAÍDA
# -----------------------------------------------------------------------------

def salvar_txt(conteudo: str, caminho_csv: str, pasta_saida: str = "output") -> str:
    os.makedirs(pasta_saida, exist_ok=True)

    nome_arquivo = os.path.basename(caminho_csv)
    nome_base, _ = os.path.splitext(nome_arquivo)

    caminho = os.path.join(pasta_saida, f"{nome_base}_relatorio.txt")

    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write(conteudo)

    return caminho


# -----------------------------------------------------------------------------
# ORQUESTRAÇÃO
# -----------------------------------------------------------------------------

def processar_pasta(pasta_dados: str = "data") -> None:
    arquivos_csv = [
        os.path.join(pasta_dados, arquivo)
        for arquivo in os.listdir(pasta_dados)
        if arquivo.endswith(".csv")
    ]

    for caminho_csv in arquivos_csv:
        print(f"\nProcessando: {caminho_csv}")
        print("-" * 40)

        df = carregar_dataset(caminho_csv)
        coluna_id = verificar_id(df)        # detecta o ID
        relatorio = info_basicas(df, coluna_id)  # passa para o relatório

        print(relatorio)

        caminho = salvar_txt(relatorio, caminho_csv)
        print(f"\nRelatório salvo em: {caminho}")


# -----------------------------------------------------------------------------
# PONTO DE ENTRADA
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    processar_pasta()
