from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .config import load_config
from .dataset.loader import load_dataset
from .dataset.validation import validate_dataset
from .export.pytorch3d_dataset import export_pytorch3d
from .preprocessing.pipeline import preprocess_dataset
from .reporting import create_dataset_report, summarize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openscan-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate the configured OpenScan dataset")
    validate.add_argument("--dataset", type=Path, help="Override the dataset environment variable")
    validate.add_argument("--config", type=Path)
    for name, help_text in (("quality","Analyze image quality"),("preprocess","Generate masks and processed images"),("export","Export a PyTorch3D-ready dataset"),("report","Generate a dataset report"),("all","Run validation, quality, preprocessing, export, and available comparison")):
        command=subparsers.add_parser(name,help=help_text); command.add_argument("--dataset",type=Path); command.add_argument("--config",type=Path)
        if name in {"preprocess","export","all"}:
            command.add_argument("--full-resolution",action="store_true",help="Keep the configured crop but do not resize it")
    compare=subparsers.add_parser("compare",help="Compare configured reference and reconstruction meshes")
    compare.add_argument("--reference",type=Path); compare.add_argument("--reconstruction",type=Path); compare.add_argument("--config",type=Path)
    summary=subparsers.add_parser("summarize",help="Summarize output folders containing completed evaluations"); summary.add_argument("root",type=Path)
    return parser


def _context(args):
    config=load_config(getattr(args,"config",None)); dataset=load_dataset(config,getattr(args,"dataset",None)); output=Path(config["dataset"]["output_dir"]).resolve(); output.mkdir(parents=True,exist_ok=True)
    if getattr(args,"full_resolution",False):
        config["crop"]["resize"]["enabled"]=False
    return config,dataset,output


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(message)s",datefmt="%H:%M:%S")
    args = build_parser().parse_args(argv)
    if args.command in {"validate","quality","preprocess","export","report","all"}:
        try:
            config,dataset,output=_context(args)
            logging.info("数据集：%s（%d 张图片）",dataset.root,len(dataset.frames))
            report = validate_dataset(dataset, config)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"Validation could not run: {exc}")
            return 2
        if args.command=="validate":
            print(json.dumps(report, indent=2)); print(f"Validation {'passed' if report['valid'] else 'failed'}"); return 0 if report["valid"] else 1
        if not report["valid"]: print("Dataset validation failed; processing stopped."); return 1
        if args.command=="quality":
            logging.info("开始裁剪、分割和对象区域质量分析")
            preprocess_dataset(dataset,config,output,write_outputs=False)
            logging.info("质量分析完成")
            print(f"Quality report: {output/'quality_report.csv'}"); return 0
        if args.command in {"preprocess","all"}:
            logging.info("开始裁剪、分割、对象质量分析和图片预处理%s","（裁剪后不缩放）" if getattr(args,"full_resolution",False) else "")
            result=preprocess_dataset(dataset,config,output)
            quality=result["quality"]
            logging.info("图片预处理完成")
        if args.command=="preprocess": print(f"Processed data: {output/'processed'}"); return 0
        if args.command in {"export","all"}:
            if args.command=="export":
                logging.info("重新生成与当前配置一致的 RGBA")
                result=preprocess_dataset(dataset,config,output);quality=result["quality"]
            logging.info("开始导出 PyTorch3D 数据集")
            export_pytorch3d(dataset,output,quality)
            logging.info("PyTorch3D 数据集导出完成")
        if args.command=="export": print(f"Export: {output/'exports'/'pytorch3d'}"); return 0
        if args.command=="all":
            ref=os.environ.get(config["mesh_comparison"]["reference_mesh_env_var"]); rec=os.environ.get("OPENSCAN_RECONSTRUCTION_MESH")
            if ref and rec and Path(ref).is_file() and Path(rec).is_file():
                from .evaluation.report import compare_meshes
                compare_meshes(ref,rec,output/"evaluation",config)
            else: print("Mesh comparison skipped: OPENSCAN_RECONSTRUCTION_MESH is not set to an existing file.")
        create_dataset_report(output); print(f"Completed output: {output}"); return 0
    if args.command=="compare":
        from .evaluation.report import compare_meshes
        config=load_config(args.config); ref=args.reference or os.environ.get(config["mesh_comparison"]["reference_mesh_env_var"]); rec=args.reconstruction or os.environ.get("OPENSCAN_RECONSTRUCTION_MESH")
        if not ref or not rec: print("Reference and reconstruction mesh paths are required."); return 2
        metrics=compare_meshes(ref,rec,Path(config["dataset"]["output_dir"]).resolve()/"evaluation",config); print(json.dumps(metrics,indent=2)); return 0
    if args.command=="summarize":
        target,count=summarize(args.root); print(f"Summarized {count} completed evaluations: {target}"); return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
