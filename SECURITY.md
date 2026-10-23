# Política de seguridad

Es un proyecto de cálculo numérico sin red ni datos de usuarios, pero si encuentras un fallo (por ejemplo, una lectura fuera de límites con ciertos parámetros), no abras un issue público: usa el aviso privado de GitHub en **Security → Report a vulnerability**. Intentaré responder en unos 7 días.

En cada cambio se compila con avisos como errores en GCC y Clang, se pasan los tests y se ejecutan gitleaks, semgrep y osv-scanner.
