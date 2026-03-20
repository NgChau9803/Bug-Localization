"""
Diagnostic script: analyze why files fail to parse in graph_builder.
Run: .venv\Scripts\activate; python scripts/diagnose_parse.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import javalang
import config
from data.data_loader import build_file_index, read_java_file

def diagnose_project(project_name, max_files=200):
    proj_cfg = config.PROJECTS[project_name]
    fi = build_file_index(proj_cfg["source_dir"])
    paths = list(fi.values())[:max_files]

    stats = {
        "total": 0,
        "parse_ok_with_methods": 0,
        "parse_ok_no_methods": 0,
        "parse_error": 0,
        "empty_file": 0,
        "errors": {},
    }

    for p in paths:
        stats["total"] += 1
        code = read_java_file(p)

        if not code or not code.strip():
            stats["empty_file"] += 1
            continue

        try:
            tree = javalang.parse.parse(code)

            # Check if it has methods
            has_methods = False
            for _, node in tree.filter(javalang.tree.MethodDeclaration):
                if node.body:
                    has_methods = True
                    break
            if not has_methods:
                for _, node in tree.filter(javalang.tree.ConstructorDeclaration):
                    if node.body:
                        has_methods = True
                        break

            if has_methods:
                stats["parse_ok_with_methods"] += 1
            else:
                stats["parse_ok_no_methods"] += 1

                # Check what it does have
                has_class = any(True for _ in tree.filter(javalang.tree.ClassDeclaration))
                has_interface = any(True for _ in tree.filter(javalang.tree.InterfaceDeclaration))
                has_enum = any(True for _ in tree.filter(javalang.tree.EnumDeclaration))
                has_annotation = any(True for _ in tree.filter(javalang.tree.AnnotationDeclaration))

                if has_annotation:
                    label = "annotation_decl"
                elif has_interface:
                    label = "interface_only"
                elif has_enum:
                    label = "enum_only"
                elif has_class:
                    label = "class_no_method_body"
                else:
                    label = "other"
                stats["errors"][label] = stats["errors"].get(label, 0) + 1

        except javalang.parser.JavaSyntaxError as e:
            stats["parse_error"] += 1
            err_type = "JavaSyntaxError"
            stats["errors"][err_type] = stats["errors"].get(err_type, 0) + 1
        except javalang.tokenizer.LexerError as e:
            stats["parse_error"] += 1
            err_type = "LexerError"
            stats["errors"][err_type] = stats["errors"].get(err_type, 0) + 1
        except Exception as e:
            stats["parse_error"] += 1
            err_type = type(e).__name__
            stats["errors"][err_type] = stats["errors"].get(err_type, 0) + 1

    return stats

if __name__ == "__main__":
    for project in config.PROJECTS:
        print(f"\n{'='*60}")
        print(f"Diagnosing: {project}")
        print(f"{'='*60}")
        s = diagnose_project(project, max_files=300)
        total = s["total"]
        print(f"  Total files sampled:      {total}")
        print(f"  Parse OK + methods:       {s['parse_ok_with_methods']} ({s['parse_ok_with_methods']/total*100:.1f}%)")
        print(f"  Parse OK, no methods:     {s['parse_ok_no_methods']} ({s['parse_ok_no_methods']/total*100:.1f}%)")
        print(f"  Parse errors:             {s['parse_error']} ({s['parse_error']/total*100:.1f}%)")
        print(f"  Empty files:              {s['empty_file']}")
        print(f"  Breakdown of non-method:")
        for k, v in sorted(s["errors"].items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
