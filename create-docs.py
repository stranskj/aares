import freephil_docs as phildoc

import os
import re
import tomllib
import importlib
import inspect
import ast

import aares


class JobClassFinder(ast.NodeVisitor):
    """AST visitor to find classes inheriting from 'Job', even with module imports."""
    def __init__(self, job_module_name="aares.Job"):
        self.job_module_name = job_module_name
        self.job_classes = []
        self.import_aliases = {}

    def visit_Import(self, node):
        """Capture imports like 'import my_project.Job as JobAlias'."""
        for alias in node.names:
            if alias.name == self.job_module_name:
                self.import_aliases[alias.asname or alias.name.split(".")[-1]] = self.job_module_name

    def visit_ImportFrom(self, node):
        """Capture imports like 'from my_project import Job'."""
        if node.module == ".".join(self.job_module_name.split(".")[:-1]):
            for alias in node.names:
                if alias.name == self.job_module_name.split(".")[-1]:
                    self.import_aliases[alias.asname or alias.name] = self.job_module_name

    def visit_ClassDef(self, node):
        """Detect classes inheriting from 'Job', considering import aliases."""
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in self.import_aliases:
                if self.import_aliases[base.id] == self.job_module_name:
                    self.job_classes.append(node.name)
            elif isinstance(base, ast.Attribute):
                # Handle cases like 'my_project.Job'
                full_name = f"{base.value.id}.{base.attr}" if isinstance(base.value, ast.Name) else None
                if full_name == self.job_module_name:
                    self.job_classes.append(node.name)
        self.generic_visit(node)


def find_descendant_classes(file_path, job_module_name="aares.Job"):
    """Find classes in the file that inherit from the base class 'Job'."""
    with open(file_path, "r") as f:
        tree = ast.parse(f.read(), filename=file_path)

    finder = JobClassFinder(job_module_name)
    finder.visit(tree)
    return finder.job_classes


def extract_functions_from_pyproject(pyproject_path,  job_module_name="aares.Job"):
    # Parse the pyproject.toml file
    with open(pyproject_path, "rb") as file:
        pyproject_data = tomllib.load(file)

    # Get the scripts section
    scripts = pyproject_data['project']['scripts']
    if not scripts:
        raise ValueError("No scripts section found in pyproject.toml.")

    results = {}

    for script_name, entry_point in scripts.items():
        module_name, _ = entry_point.split(":")
        module_path = module_name.replace(".", os.sep) + ".py"

        if os.path.exists(module_path):
            # Step 1: Detect descendant classes statically
            descendant_classes = find_descendant_classes(module_path, job_module_name)

            # Step 2: Dynamically import and retrieve class objects
            class_objects = []
            try:
                module = importlib.import_module(module_name)
                for cls_name in descendant_classes:
                    cls = getattr(module, cls_name, None)
                    if cls and inspect.isclass(cls):
                        # Verify if the class inherits from 'Job'
                        base_classes = [base.__name__ for base in inspect.getmro(cls)]
                        if "Job" in base_classes:
                            class_objects.append(cls)
                        else:
                            print(f"Class {cls_name} in {module_name} is not a descendant of Job.")
                    else:
                        print(f"Class {cls_name} not found in module {module_name}.")
            except ImportError as e:
                print(f"Error importing module {module_name}: {e}")

            # Store the class objects in the results
            if class_objects:
                results[script_name] = {
                    "module_name": module_name,
                    "classes": class_objects,  # List of class objects
                }
        else:
            print(f"Warning: {module_path} does not exist.")

    return results


def create_docs(pyproject_path = "pyproject.toml", job_module_name="aares.Job", out_path = "wiki"):

    extracted = extract_functions_from_pyproject(pyproject_path, job_module_name)

    aares.create_directory(out_path)

    with open(os.path.join(out_path,"List_of_tools.md"),'w') as flist:
        flist.write('''
AAres tools
===========
       
''')

        for script_name, data in extracted.items():
            for cls in data["classes"]:
                print(script_name)
                flist.write(f'* **{script_name}**: {cls.short_description}\n')



if __name__ == "__main__":
    create_docs(out_path='test-wiki')

    pass