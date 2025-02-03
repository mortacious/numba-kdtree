from setuptools import setup, Extension
from distutils.command.build_ext import build_ext as build_ext_orig
from distutils.sysconfig import get_python_inc
from os.path import join, dirname
from distutils.sysconfig import customize_compiler
import platform
import numba.extending as nbe

inc_dirs = [get_python_inc(), nbe.include_path()]
inc_dirs.append(join(dirname(dirname(__file__)), 'src', 'ckdtree'))

ckdtree_src = ['init.cpp',
               'build.cpp',
               'query.cpp',
               'query_radius.cpp'
               ]

ckdtree_src = [join('src', 'ckdtree', x) for x in ckdtree_src]


class CTypesExtension(Extension): pass


class build_ext(build_ext_orig):
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        if self._ctypes:
            customize_compiler(self.compiler)
            try:
                self.compiler.compiler_so.remove("-Wstrict-prototypes")
            except (AttributeError, ValueError):
                pass
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            if platform.system() == "Windows":
                return ext_name + '.lib'
            else:
                return ext_name + '.so'
        return super().get_ext_filename(ext_name)

if platform.system() == "Windows":
    extra_compile_args = ['/O2', '/DKDTREE_COMPILING=1']
else:
    extra_compile_args = ['-fPIC', '-shared', '-O3', '-DKDTREE_COMPILING=1']
    if platform.system() == "Darwin":
        extra_compile_args.append('-std=c++11')

module = CTypesExtension('numba_kdtree._ckdtree',
                   sources=ckdtree_src,
                   include_dirs=inc_dirs,
                   extra_compile_args=extra_compile_args)

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[module],
    zip_safe=True,
)
