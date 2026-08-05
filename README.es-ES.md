

# AmpHGT

**AmpHGT: Expansión de la Predicción de Actividad Antimicrobiana en Péptidos que Contienen Aminoácidos No Canónicos Mediante un Transformador de Grafos Heterogéneos con Restricciones Multivista**

Este repositorio contiene el código del artículo y proporciona todas las interfaces necesarias para el entrenamiento e inferencia del modelo.

---

- **Script principal:**
  `main.py` proporciona la interfaz para el entrenamiento e inferencia del modelo.

- **Datos de entrenamiento:**
  Los datos de entrenamiento utilizados en el artículo se pueden descargar desde [AmpHGT_db](https://github.com/AledHe/AmpHGT_db).

- **Opciones de configuración:**
  Puede modificar los archivos de configuración YAML en el directorio `configs/` o anular los parámetros mediante argumentos de línea de comandos.

---

## Configuración del Entorno

Se recomienda crear un nuevo entorno conda con nuestro environment.yml mediante:

```bash
conda env create -f environment.yml
```

## Entrenamiento del Modelo

Existen dos enfoques principales para el entrenamiento:

1. **Uso de archivos de configuración YAML:**
   Modifique los parámetros en los archivos YAML ubicados en el directorio `configs/`.

2. **Uso de argumentos de línea de comandos:**
   Proporcione los parámetros detallados directamente al ejecutar el comando.

### Inicio Rápido para el Entrenamiento

Para comenzar el entrenamiento con los parámetros predeterminados (definidos en `configs/finetune_binary.yaml`), simplemente ejecute:

```bash
python main.py ftb
```

> **Nota:**
> A pesar del nombre `finetune_binary`, este modo no utiliza ningún PharmHGT preentrenado, ya que `load_pretrained` está configurado en `False` de forma predeterminada.

### Comando Recomendado para Reproducir Resultados

Para una configuración más controlada y reproducir nuestros resultados reportados, intente el siguiente comando:

```bash
python main.py ftb -c configs/finetune_binary.yaml -o out_finetune_binary/gru_npt \
  -train*readout gru -train*fusion attention -train*sq_embed ESM2 \
  -train*decay 1e-2 -train*patience 10 -train*seed 512
```

- `-c`: Especifica qué archivo de configuración usar.
- `-o`: Especifica el directorio de salida para la ejecución.
- Todos los demás parámetros (como `-train*readout`, `-train*fusion`, etc.) anulan la configuración correspondiente en el archivo YAML.

### Atajos de Configuración

- **`ft`** usa `configs/pretrain.yaml` y guarda la salida en `out_pretrain`.
- **`ftb`** usa `configs/finetune_binary.yaml` y guarda la salida en `out_finetune_binary`.
- **`ifb`** usa `configs/inference_binary.yaml` y guarda la salida en `out_inference_binary`.

---

## Inferencia

Para realizar la inferencia con su modelo entrenado, utilice un comando similar al siguiente:

```bash
python main.py ifb -o out_test -train*readout gru -train*fusion attention \
  -train*sq_embed ESM2 -train*checkpoint_path your/model/path/model.pt -train*batch_size 512
```

- Reemplace `your/model/path/model.pt` por la ruta real a su checkpoint del modelo.
- La opción `-o` define el directorio de salida para la ejecución de inferencia.

### Nota Adicional sobre el Procesamiento de Datos

Si se proporciona un nuevo archivo `.smi` (datos SMILES sin procesar), el modelo preprocesará automáticamente el archivo en grafos guardados por DGL en la carpeta `tmp/`. Permita un tiempo de procesamiento adicional para este paso.
