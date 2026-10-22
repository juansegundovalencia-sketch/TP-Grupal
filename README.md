# TP grupal — ALC 2C2026

Laboratorios 3 y 6 de Álgebra Lineal Computacional.

## Contenido

| Archivo | Descripción |
|---|---|
| `labo3.py` | Normas vectoriales y matriciales (`norma`, `normaliza`, `normaMatMC`, `normaExacta`, `condMC`). Incluye sus propios tests al importarse. |
| `Labo06.py` | Método de la potencia (`metpot2k`) y diagonalización por reflexiones de Householder (`diagRH`). Importa `labo3`. |
| `test_L06.py` | Tests del Labo 6. |

## Requisitos

```bash
pip install numpy matplotlib
```

## Correr los tests

```bash
python3 test_L06.py
```

Si no imprime ningún `AssertionError`, pasaron todos.

## Cómo colaborar

```bash
git clone <url-del-repo>
cd TP-grupal
git checkout -b mi-rama
# ... cambios ...
git add -A && git commit -m "descripción del cambio"
git push -u origin mi-rama
```

Después abrir un Pull Request desde la web. Evitemos pushear directo a `main`
para no pisarnos entre nosotros.
