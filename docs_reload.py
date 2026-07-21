# import sys
# import griffe


# class FreshImport(griffe.Extension):
#     """Evict the package from sys.modules after each load so `mkdocs serve`
#     re-imports edited source on the next in-process rebuild."""

#     def __init__(self, package="popinn"):
#         self.package = package

#     def on_package(self, *, pkg, **kw):
#         for name in list(sys.modules):
#             if name == self.package or name.startswith(self.package + "."):
#                 del sys.modules[name]


# def _field_default(value):
#     """If `value` is an `eqx.field(..., default=X)` call expression, return X."""
#     if isinstance(value, griffe.ExprCall):
#         for arg in value.arguments:
#             if isinstance(arg, griffe.ExprKeyword) and arg.name == "default":
#                 return arg.value
#     return None


# class StaticDefaults(griffe.Extension):
#     """Replace inspected parameter defaults with the statically parsed
#     (source-text) versions, so the table's Default column shows values as
#     written (e.g. jnp.tanh) instead of runtime repr()s."""

#     def __init__(self):
#         self._static = {}

#     def _pkg(self, name, loader):
#         if name not in self._static:
#             self._static[name] = griffe.load(
#                 name,
#                 search_paths=[str(p) for p in loader.finder.search_paths],
#                 force_inspection=False,
#             )
#         return self._static[name]

#     def on_function(self, *, func, loader, **kw):
#         pkg = self._pkg(func.package.name, loader)
#         rel = func.path[len(func.package.name) + 1:]
#         try:
#             static = pkg[rel]
#         except KeyError:
#             static = None
#         if isinstance(static, griffe.Function):
#             defaults = {p.name: p.default for p in static.parameters}
#         elif func.name == "__init__":            # equinox-generated __init__
#             try:
#                 cls = pkg[rel.rsplit(".", 1)[0]]
#             except KeyError:
#                 return
#             defaults = {n: _field_default(getattr(m, "value", None)) for n, m in cls.members.items()}
#         else:
#             return
#         for p in func.parameters:
#             if defaults.get(p.name) is not None:
#                 p.default = defaults[p.name]


# docs_reload.py  (repo root, next to mkdocs.yml)
import sys

import griffe


class FreshImport(griffe.Extension):
    def __init__(self, package="popinn"):
        self.package = package

    def on_package(self, *, pkg, **kw):
        for name in list(sys.modules):
            if name == self.package or name.startswith(self.package + "."):
                del sys.modules[name]


def _field_default(value):
    """If value is an eqx.field(..., default=X) call expression, return X."""
    if isinstance(value, griffe.ExprCall):
        for arg in value.arguments:
            if isinstance(arg, griffe.ExprKeyword) and arg.name == "default":
                return arg.value
    return None


class StaticDefaults(griffe.Extension):
    """force_inspection records defaults/values as runtime repr()s. Overwrite
    them with the statically parsed (source-text) versions, so they render as
    written and need no upkeep when the values change."""

    def __init__(self):
        self._static = {}

    def _pkg(self, name, loader):
        if name not in self._static:
            self._static[name] = griffe.load(
                name,
                search_paths=[str(p) for p in loader.finder.search_paths],
                force_inspection=False,
            )
        return self._static[name]

    def _lookup(self, obj, loader):
        pkg = self._pkg(obj.package.name, loader)
        rel = obj.path[len(obj.package.name) + 1 :]
        try:
            return pkg, rel, pkg[rel]
        except KeyError:
            return pkg, rel, None

    def on_function(self, *, func, loader, **kw):
        pkg, rel, static = self._lookup(func, loader)
        if isinstance(static, griffe.Function):
            defaults = {p.name: p.default for p in static.parameters}
        elif func.name == "__init__":  # eqx-generated __init__
            try:
                cls = pkg[rel.rsplit(".", 1)[0]]
            except KeyError:
                return
            defaults = {n: _field_default(getattr(m, "value", None)) for n, m in cls.members.items()}
        else:
            return
        for p in func.parameters:
            if defaults.get(p.name) is not None:
                p.default = defaults[p.name]

    def on_attribute(self, *, attr, loader, **kw):
        _, _, static = self._lookup(attr, loader)
        if isinstance(static, griffe.Attribute):
            attr.value = static.value
