"""
main.py
=======
Orquestador general del proyecto. NO contiene lógica propia: solo
ofrece una CLI única con subcomandos que delegan en los scripts de
`scripts/` (que a su vez delegan en los módulos de `src/`). Se ofrece
por comodidad; cada paso también puede ejecutarse por separado.

Uso:
    python main.py build-corpus --n-queries 40
    python main.py index
    python main.py evaluate
    python main.py chat        # imprime el comando para lanzar la UI
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sistema RAG Multimodal - orquestador")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-corpus", help="Literal a) preparar el corpus")
    p_build.add_argument("--n-queries", type=int, default=40)
    p_build.add_argument("--no-images", action="store_true")

    sub.add_parser("index", help="Literales b) y c): embeddings + ChromaDB")
    sub.add_parser("evaluate", help="Literal f): Precision@k, Recall@k, NDCG@k")
    sub.add_parser("chat", help="Literal e): lanzar la interfaz web (Streamlit)")
    sub.add_parser("all", help="Ejecuta build-corpus + index + evaluate en secuencia")

    args = parser.parse_args()
    python = sys.executable

    if args.command == "build-corpus":
        cmd = [python, "scripts/build_corpus.py", "--n-queries", str(args.n_queries)]
        if args.no_images:
            cmd.append("--no-images")
        run(cmd)
    elif args.command == "index":
        run([python, "scripts/index_corpus.py"])
    elif args.command == "evaluate":
        run([python, "scripts/run_evaluation.py"])
    elif args.command == "chat":
        print("Ejecuta en tu terminal:\n\n    streamlit run app.py\n")
    elif args.command == "all":
        run([python, "scripts/build_corpus.py"])
        run([python, "scripts/index_corpus.py"])
        run([python, "scripts/run_evaluation.py"])


if __name__ == "__main__":
    main()