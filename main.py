import yaml
import pandas as pd
from sklearn.datasets import load_wine
import sys
from datetime import datetime


def load_config(config_path: str) -> dict:
    """Lê o arquivo YAML de configuração e retorna um dicionário com os parâmetros."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(csv_path: str, config: dict) -> pd.DataFrame:
    """Carrega o CSV do dataset usando as configurações do YAML (separador e encoding)."""
    df = pd.read_csv(
        csv_path,
        sep=config["dataset"]["separator"],
        encoding=config["dataset"]["encoding"]
    )
    return df


def prepare_wine_dataset():
    """
    Carrega o dataset Wine do scikit-learn e o salva como CSV na pasta data/.
    Adiciona uma coluna de ID e renomeia o target para 'diagnostico',
    simulando o formato esperado dos datasets reais da IC.
    """
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df.insert(0, "id", range(1, len(df) + 1))  # Coluna de ID começando em 1
    df["diagnostico"] = wine.target
    df.to_csv("data/dataset.csv", index=False)
    print("Dataset Wine salvo em data/dataset.csv!")


def get_feature_cols(df: pd.DataFrame, config: dict) -> list:
    """
    Retorna apenas as colunas de características, excluindo ID e classe(s).
    Essa lógica é repetida em várias funções, então centralizamos aqui.
    """
    id_cols = [config["dataset"]["id_column"]] if config["dataset"]["has_id_column"] else []
    class_cols = config["dataset"]["class_columns"]
    return [c for c in df.columns if c not in id_cols and c not in class_cols]


def print_basic_info(df: pd.DataFrame, config: dict) -> None:
    """Imprime o número de instâncias, características e colunas de classe do dataset."""
    feature_cols = get_feature_cols(df, config)

    print("=== INFORMAÇÕES BÁSICAS ===")
    print(f"Instâncias: {len(df)}")
    print(f"Características: {len(feature_cols)}")
    print(f"Colunas de classe: {config['dataset']['class_columns']}")


def print_missing_values(df: pd.DataFrame, config: dict) -> None:
    """
    Analisa missing values (valores ausentes) no dataset.
    Imprime quantas instâncias possuem ao menos um missing value,
    e quais características têm missing values e em que proporção.
    """
    feature_cols = get_feature_cols(df, config)
    df_features = df[feature_cols]
    total = len(df)

    # Verifica quais linhas têm ao menos um valor ausente
    instances_with_missing = df_features.isnull().any(axis=1).sum()

    print("\n=== MISSING VALUES ===")
    print(f"Instâncias com algum missing value: {instances_with_missing} de {total} ({100 * instances_with_missing / total:.1f}%)")

    # Conta missing values por coluna
    missing_by_col = df_features.isnull().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0]

    if len(cols_with_missing) == 0:
        print("Nenhuma característica possui missing values.")
    else:
        print("Características com missing values:")
        for col, count in cols_with_missing.items():
            print(f"  {col}: {count} ({100 * count / total:.1f}%)")


def print_zero_values(df: pd.DataFrame, config: dict) -> None:
    """
    Analisa valores zero no dataset.
    Valores zero em medidas faciais podem indicar falha na detecção
    de pontos de controle, sendo importante identificá-los.
    Imprime cada característica com zeros, sua contagem e percentual.
    """
    feature_cols = get_feature_cols(df, config)
    df_features = df[feature_cols]
    total = len(df)

    print("\n=== VALORES ZERO ===")
    zeros_by_col = (df_features == 0).sum()
    cols_with_zeros = zeros_by_col[zeros_by_col > 0]

    if len(cols_with_zeros) == 0:
        print("Nenhuma característica possui valores zero.")
    else:
        # Imprime tabela formatada com alinhamento de colunas
        print(f"{'Característica':<40} {'Zeros':>6} {'%':>8}")
        print("-" * 56)
        for col, count in cols_with_zeros.items():
            print(f"{col:<40} {count:>6} {100 * count / total:>7.1f}%")


def print_duplicates(df: pd.DataFrame, config: dict) -> None:
    """
    Verifica instâncias duplicadas no dataset (ignorando ID e classe).
    Duplicatas podem distorcer o treinamento e a avaliação dos classificadores.
    Se encontradas, imprime os índices das linhas duplicadas.
    """
    feature_cols = get_feature_cols(df, config)
    df_features = df[feature_cols]
    total = len(df)

    duplicates = df_features.duplicated().sum()

    print("\n=== INSTÂNCIAS DUPLICADAS ===")
    if duplicates == 0:
        print("Nenhuma instância duplicada encontrada.")
    else:
        print(f"Instâncias duplicadas: {duplicates} de {total} ({100 * duplicates / total:.1f}%)")
        print("Índices das duplicatas:")
        # keep='first' mantém a primeira ocorrência e marca as demais como duplicatas
        print(df[df_features.duplicated(keep='first')].index.tolist())


def setup_output(output_dir: str):
    """
    Configura o salvamento do relatório em arquivo texto na pasta output/.
    Usa a classe Tee para redirecionar os prints simultaneamente
    para o terminal e para o arquivo de log.
    O nome do arquivo inclui o timestamp para evitar sobrescritas.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)  # Cria a pasta se não existir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{output_dir}/relatorio_{timestamp}.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)  # Substitui o stdout pelo Tee
    print(f"Relatório gerado em: {log_path}")
    return log_file


class Tee:
    """
    Redireciona o print para múltiplos streams ao mesmo tempo.
    Usada para escrever simultaneamente no terminal e no arquivo de log,
    sem precisar chamar print() duas vezes.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        """Escreve o conteúdo em todos os streams registrados."""
        for s in self.streams:
            s.write(data)

    def flush(self):
        """
        Força o envio dos dados em buffer para todos os streams.
        O try/except ignora o erro caso o arquivo já tenha sido fechado
        (o que pode ocorrer no encerramento do programa).
        """
        for s in self.streams:
            try:
                s.flush()
            except ValueError:
                pass


if __name__ == "__main__":
    prepare_wine_dataset()

    config = load_config("config/config.yaml")
    df = load_dataset("data/dataset.csv", config)

    # Inicia o log — tudo que for impresso a partir daqui vai para o arquivo também
    log = setup_output(config["output"]["output_dir"])

    print(df.head())
    print_basic_info(df, config)
    print_missing_values(df, config)
    print_zero_values(df, config)
    print_duplicates(df, config)

    log.close()  # Fecha o arquivo de log ao final