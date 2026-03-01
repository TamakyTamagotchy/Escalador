# ui_pruebas.py
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading
import shutil
import logging
from queue import Queue, Empty
from pathlib import Path
from prueba_config import detectar_upscalers_instalados
from config.models_config import UPSCALERS, construir_comando
from config.Process.video_converter import convertir_frames_a_video
from config.Process.image_processor import mejorar_imagen
from config.Process.frame_processor import mejorar_frames, extraer_frames

logger = logging.getLogger(__name__)

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    using_dnd = True
except ImportError:
    using_dnd = False

class AplicacionMejoradorVideo(ctk.CTk if not using_dnd else TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mejorador de Video e Imagen IA Pro")
        self.geometry("850x750")
        self._setup_inicial()
        self._crear_interfaz()
        self.queue = Queue()
        self.processing_thread = None
        self.is_processing = False

        # Habilitar redimensionamiento
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.panel_principal.grid(row=0, column=0, sticky="nsew")

    def run_in_background(self, target, args=(), callback=None):
        """Ejecuta una funcion en segundo plano"""
        def wrapper():
            try:
                result = target(*args)
                if callback:
                    self.queue.put((callback, result))
            except Exception as e:
                self.queue.put((self._handle_error, str(e)))
            finally:
                self.queue.put((self._end_processing, None))

        self.processing_thread = threading.Thread(target=wrapper)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self._check_queue()

    def _check_queue(self):
        """Verifica la cola de mensajes del hilo secundario"""
        try:
            while True:
                callback, args = self.queue.get_nowait()
                if args is not None:
                    callback(args)
                else:
                    try:
                        callback()
                    except TypeError:
                        callback(None)
        except Empty:
            pass
        except Exception as e:
            logger.error(f"Error en _check_queue: {e}")
        finally:
            if self.is_processing:
                self.after(100, self._check_queue)

    def _update_status(self, text):
        """Thread-safe status label update via queue."""
        self.queue.put((lambda t=text: self.label_estado.configure(text=t), None))

    def _update_progress(self, value):
        """Thread-safe progress bar update via queue."""
        self.queue.put((lambda v=value: self.barra_progreso.set(v), None))

    def _handle_error(self, error_msg):
        """Maneja errores del proceso en segundo plano"""
        self.label_estado.configure(text="Error en el proceso")
        messagebox.showerror("Error", error_msg)
        self._configurar_interfaz_proceso(True)
        self.is_processing = False

    def _end_processing(self):
        """Finaliza el procesamiento"""
        self.is_processing = False
        self._configurar_interfaz_proceso(True)

    def _setup_inicial(self):
        # Configuracion del tema
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color="#1a1a1a" if not using_dnd else None)

        # Variables de estado
        self.archivo_seleccionado = None

        # Detectar upscalers instalados
        self.upscalers_instalados = detectar_upscalers_instalados()

        if not self.upscalers_instalados:
            messagebox.showerror(
                "Error",
                "No se encontro ningun upscaler instalado.\n"
                "Asegurese de que exista al menos realesrgan-ncnn-vulkan.exe\n"
                "en la carpeta bin/ del proyecto con su carpeta models/."
            )
            self.destroy()
            return

        # Variables de control
        self.solo_imagenes = ctk.BooleanVar(value=True)
        primer_upscaler_id = list(self.upscalers_instalados.keys())[0]
        self.upscaler_var = ctk.StringVar(value=primer_upscaler_id)
        primer_upscaler = UPSCALERS[primer_upscaler_id]
        primer_modelo_id = list(primer_upscaler.modelos.keys())[0]
        self.modelo_var = ctk.StringVar(value=primer_modelo_id)
        primer_modelo = primer_upscaler.modelos[primer_modelo_id]
        self.var_escala = ctk.IntVar(value=primer_modelo.escala_predeterminada)
        self.var_denoise = ctk.IntVar(value=-1)

        # Configuracion DnD si esta disponible
        if using_dnd:
            self.drop_target_register(DND_FILES)
            self.dnd_bind('<<Drop>>', self._procesar_archivo_entrada)

    def _crear_interfaz(self):
        # Panel principal
        self.panel_principal = ctk.CTkFrame(self)
        self.panel_principal.pack(fill="both", expand=True, padx=20, pady=20)

        # Habilitar redimensionamiento en panel principal
        self.panel_principal.grid_rowconfigure(0, weight=1)
        self.panel_principal.grid_columnconfigure(0, weight=1)

        # Cabecera
        self._crear_cabecera()

        # Area principal
        self._crear_area_principal()

        # Panel de control
        self._crear_panel_control()

        # Barra de estado
        self._crear_barra_estado()

    def _crear_cabecera(self):
        frame_cabecera = ctk.CTkFrame(self.panel_principal)
        frame_cabecera.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkLabel(
            frame_cabecera,
            text="Mejorador de Video e Imagen IA Pro",
            font=("Roboto", 24, "bold")
        ).pack(side="left", pady=10, padx=10)

        ctk.CTkButton(
            frame_cabecera,
            text="?",
            width=30,
            command=self._mostrar_ayuda
        ).pack(side="right", pady=10, padx=10)

    def _crear_area_principal(self):
        frame_principal = ctk.CTkFrame(self.panel_principal)
        frame_principal.pack(fill="both", expand=True, padx=10, pady=10)

        frame_principal.grid_rowconfigure(0, weight=1)
        frame_principal.grid_columnconfigure(0, weight=1)

        self._crear_selector_archivo(frame_principal)
        self._crear_vista_previa(frame_principal)

    def _crear_selector_archivo(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=5, pady=5)

        self.label_archivo = ctk.CTkLabel(
            frame,
            text="Arrastra y suelta un archivo o selecciona",
            font=("Roboto", 12)
        )
        self.label_archivo.pack(side="left", padx=10)

        frame_botones = ctk.CTkFrame(frame)
        frame_botones.pack(side="right", padx=5)

        self.boton_seleccionar = ctk.CTkButton(
            frame_botones,
            text="Seleccionar Archivo",
            command=self._seleccionar_archivo,
            width=150
        )
        self.boton_seleccionar.pack(side="left", padx=5)

        ctk.CTkButton(
            frame_botones,
            text="Limpiar",
            command=self._limpiar_seleccion,
            width=100
        ).pack(side="left", padx=5)

    def _seleccionar_archivo(self):
        self._procesar_archivo_entrada()

    def _crear_vista_previa(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame,
            text="Vista Previa",
            font=("Roboto", 14, "bold")
        ).pack(pady=5)

        self.canvas_preview = tk.Canvas(
            frame,
            bg="#1a1a1a",
            width=800,
            height=400,
            highlightthickness=0
        )
        self.canvas_preview.pack(fill="both", expand=True, padx=10, pady=10)

    def _crear_panel_control(self):
        frame = ctk.CTkFrame(self.panel_principal)
        frame.pack(fill="x", padx=10, pady=10)

        # Fila superior: checkbox + upscaler + modelo
        frame_fila1 = ctk.CTkFrame(frame)
        frame_fila1.pack(fill="x", padx=5, pady=(5, 2))

        ctk.CTkCheckBox(
            frame_fila1,
            text="Solo imagenes",
            variable=self.solo_imagenes,
            command=self._actualizar_interfaz
        ).pack(side="left", padx=10)

        ctk.CTkLabel(frame_fila1, text="Upscaler:").pack(side="left", padx=(20, 5))
        upscaler_nombres = []
        self._upscaler_display_map = {}
        for uid in self.upscalers_instalados:
            info = UPSCALERS[uid]
            display = info.nombre_display
            upscaler_nombres.append(display)
            self._upscaler_display_map[display] = uid

        self.selector_upscaler = ctk.CTkOptionMenu(
            frame_fila1,
            values=upscaler_nombres,
            command=self._on_upscaler_changed,
            width=180
        )
        self.selector_upscaler.pack(side="left", padx=5)
        # Set initial display value
        primer_uid = self.upscaler_var.get()
        self.selector_upscaler.set(UPSCALERS[primer_uid].nombre_display)

        ctk.CTkLabel(frame_fila1, text="Modelo:").pack(side="left", padx=(20, 5))
        self.selector_modelo = ctk.CTkOptionMenu(
            frame_fila1,
            values=[""],
            command=self._on_modelo_changed,
            width=180
        )
        self.selector_modelo.pack(side="left", padx=5)

        # Fila inferior: escala + denoise + boton
        frame_fila2 = ctk.CTkFrame(frame)
        frame_fila2.pack(fill="x", padx=5, pady=(2, 5))

        ctk.CTkLabel(frame_fila2, text="Escala:").pack(side="left", padx=(10, 5))
        self.selector_escala = ctk.CTkOptionMenu(
            frame_fila2,
            values=["2x"],
            command=lambda x: self.var_escala.set(int(x.replace("x", ""))),
            width=80
        )
        self.selector_escala.pack(side="left", padx=5)

        self.label_denoise = ctk.CTkLabel(frame_fila2, text="Ruido:")
        self.label_denoise.pack(side="left", padx=(20, 5))
        self.selector_denoise = ctk.CTkOptionMenu(
            frame_fila2,
            values=["-1 (auto)", "0", "1", "2", "3"],
            command=lambda x: self.var_denoise.set(int(x.split()[0])),
            width=100
        )
        self.selector_denoise.pack(side="left", padx=5)

        self.boton_mejorar = ctk.CTkButton(
            frame_fila2,
            text="Iniciar Mejora",
            command=self._iniciar_proceso,
            state="disabled",
            width=150
        )
        self.boton_mejorar.pack(side="right", padx=10)

        # Inicializar dropdowns con el upscaler seleccionado
        self._actualizar_modelos_dropdown()

    def _on_upscaler_changed(self, display_name):
        """Cuando el usuario cambia el upscaler."""
        upscaler_id = self._upscaler_display_map.get(display_name)
        if upscaler_id:
            self.upscaler_var.set(upscaler_id)
            self._actualizar_modelos_dropdown()

    def _on_modelo_changed(self, display_name):
        """Cuando el usuario cambia el modelo."""
        upscaler_id = self.upscaler_var.get()
        upscaler = UPSCALERS.get(upscaler_id)
        if not upscaler:
            return

        # Encontrar modelo_id por nombre display
        paths = self.upscalers_instalados.get(upscaler_id)
        for modelo_id, modelo_info in upscaler.modelos.items():
            if modelo_info.nombre_display == display_name and modelo_id in paths.ruta_modelos:
                self.modelo_var.set(modelo_id)
                self._actualizar_escalas_dropdown()
                self._actualizar_denoise_visibilidad()
                return

    def _actualizar_modelos_dropdown(self):
        """Actualiza el dropdown de modelos segun el upscaler seleccionado."""
        upscaler_id = self.upscaler_var.get()
        upscaler = UPSCALERS.get(upscaler_id)
        paths = self.upscalers_instalados.get(upscaler_id)

        if not upscaler or not paths:
            return

        # Solo mostrar modelos que tienen archivos instalados
        modelos_disponibles = []
        for modelo_id, modelo_info in upscaler.modelos.items():
            if modelo_id in paths.ruta_modelos:
                modelos_disponibles.append(modelo_info.nombre_display)

        if modelos_disponibles:
            self.selector_modelo.configure(values=modelos_disponibles)
            self.selector_modelo.set(modelos_disponibles[0])
            # Sync modelo_var
            self._on_modelo_changed(modelos_disponibles[0])
        else:
            self.selector_modelo.configure(values=["Sin modelos"])
            self.selector_modelo.set("Sin modelos")

    def _actualizar_escalas_dropdown(self):
        """Actualiza el dropdown de escalas segun el modelo seleccionado."""
        upscaler_id = self.upscaler_var.get()
        modelo_id = self.modelo_var.get()
        upscaler = UPSCALERS.get(upscaler_id)

        if not upscaler:
            return

        modelo_info = upscaler.modelos.get(modelo_id)
        if not modelo_info:
            return

        escalas = modelo_info.escalas
        self.selector_escala.configure(values=[f"{x}x" for x in escalas])
        self.selector_escala.set(f"{modelo_info.escala_predeterminada}x")
        self.var_escala.set(modelo_info.escala_predeterminada)

    def _actualizar_denoise_visibilidad(self):
        """Muestra/oculta el selector de denoise segun el modelo."""
        upscaler_id = self.upscaler_var.get()
        modelo_id = self.modelo_var.get()
        upscaler = UPSCALERS.get(upscaler_id)

        if not upscaler:
            return

        modelo_info = upscaler.modelos.get(modelo_id)
        tiene_denoise = modelo_info and modelo_info.denoise_niveles is not None

        if tiene_denoise:
            niveles = modelo_info.denoise_niveles
            valores_display = []
            for n in niveles:
                if n == -1:
                    valores_display.append("-1 (auto)")
                else:
                    valores_display.append(str(n))
            self.selector_denoise.configure(values=valores_display)
            self.selector_denoise.set(valores_display[0])
            self.var_denoise.set(niveles[0])
            self.label_denoise.pack(side="left", padx=(20, 5))
            self.selector_denoise.pack(side="left", padx=5)
        else:
            self.label_denoise.pack_forget()
            self.selector_denoise.pack_forget()
            self.var_denoise.set(-1)

    def _crear_barra_estado(self):
        frame = ctk.CTkFrame(self.panel_principal)
        frame.pack(fill="x", padx=10, pady=5)

        frame_info = ctk.CTkFrame(frame)
        frame_info.pack(fill="x", padx=5, pady=5)

        self.barra_progreso = ctk.CTkProgressBar(frame_info)
        self.barra_progreso.pack(fill="x", padx=5, pady=5)
        self.barra_progreso.set(0)

        self.label_estado = ctk.CTkLabel(
            frame_info,
            text="Listo",
            font=("Roboto", 12)
        )
        self.label_estado.pack(pady=5)

        self.label_info = ctk.CTkLabel(
            frame_info,
            text="",
            font=("Roboto", 10)
        )
        self.label_info.pack(pady=2)

    def _mostrar_ayuda(self):
        # Listar upscalers instalados
        upscalers_txt = ""
        for uid, paths in self.upscalers_instalados.items():
            info = UPSCALERS[uid]
            n_modelos = len(paths.ruta_modelos)
            upscalers_txt += f"  - {info.nombre_display} ({n_modelos} modelos)\n"

        messagebox.showinfo("Ayuda",
            "Instrucciones de uso:\n\n"
            "1. Selecciona un archivo de video o imagen\n"
            "2. Elige el upscaler y modelo deseado\n"
            "3. Ajusta la escala (y nivel de ruido si aplica)\n"
            "4. Presiona 'Iniciar Mejora'\n\n"
            "Formatos soportados:\n"
            "- Videos: MP4, AVI, MKV, MOV\n"
            "- Imagenes: PNG, JPG, JPEG, BMP, WEBP\n\n"
            f"Upscalers instalados:\n{upscalers_txt}\n"
            "Para agregar mas upscalers, consulte bin/LEEME.txt"
        )

    def _limpiar_seleccion(self):
        self.archivo_seleccionado = None
        self.canvas_preview.delete("all")
        self._actualizar_interfaz()
        self.label_archivo.configure(text="Arrastra y suelta un archivo o selecciona")
        self.label_info.configure(text="")

    def _procesar_archivo_entrada(self, event=None):
        archivo = event.data.strip('{}') if event else filedialog.askopenfilename(
            filetypes=[("Archivos soportados",
                    "*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.avi *.mkv *.mov")])

        if archivo and self._validar_archivo(archivo):
            self.archivo_seleccionado = archivo
            self._actualizar_interfaz()
            self._mostrar_vista_previa()
            self._mostrar_info_archivo()

    def _validar_archivo(self, archivo):
        ext = Path(archivo).suffix.lower()
        if self.solo_imagenes.get():
            return ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
        return ext in ['.mp4', '.avi', '.mkv', '.mov']

    def _mostrar_info_archivo(self):
        if not self.archivo_seleccionado:
            return
        size = os.path.getsize(self.archivo_seleccionado) / (1024 * 1024)  # MB
        if self.solo_imagenes.get():
            img = Image.open(self.archivo_seleccionado)
            info = f"Tamano: {size:.1f}MB | Dimensiones: {img.size[0]}x{img.size[1]}"
            img.close()
        else:
            cap = cv2.VideoCapture(self.archivo_seleccionado)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames / fps if fps else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            info = f"Tamano: {size:.1f}MB | Dimensiones: {width}x{height} | FPS: {fps:.1f} | Duracion: {duration:.1f}s"
        self.label_info.configure(text=info)

    def _mostrar_vista_previa(self):
        try:
            if self.solo_imagenes.get():
                imagen = Image.open(self.archivo_seleccionado)
            else:
                cap = cv2.VideoCapture(self.archivo_seleccionado)
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("No se pudo leer el archivo de video")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imagen = Image.fromarray(frame)
                cap.release()
            # Ajustar tamano y mostrar
            w, h = imagen.size
            ratio = min(800/w, 400/h)
            imagen = imagen.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
            foto = ImageTk.PhotoImage(imagen)
            self.canvas_preview.delete("all")
            self.canvas_preview.create_image(400, 200, image=foto)
            self.canvas_preview.image = foto
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar la vista previa: {e}")

    def _obtener_rutas_actuales(self):
        """Retorna (ruta_exe, ruta_modelos) para la seleccion actual de upscaler/modelo."""
        upscaler_id = self.upscaler_var.get()
        modelo_id = self.modelo_var.get()
        paths = self.upscalers_instalados.get(upscaler_id)

        if not paths:
            return None, None

        ruta_exe = paths.ruta_exe
        ruta_modelos = paths.ruta_modelos.get(modelo_id)

        return ruta_exe, ruta_modelos

    def _procesar_video(self, directorio_salida):
        upscaler_id = self.upscaler_var.get()
        modelo_id = self.modelo_var.get()
        ruta_exe, ruta_modelos = self._obtener_rutas_actuales()

        directorio_temp = os.path.join(directorio_salida, "temp_frames")
        os.makedirs(directorio_temp, exist_ok=True)
        archivo_salida = os.path.join(
            directorio_salida,
            f"{Path(self.archivo_seleccionado).stem}_mejorado.mp4"
        )
        try:
            self._update_status("Extrayendo frames...")
            extraer_frames(self.archivo_seleccionado, directorio_temp)

            self._update_status("Mejorando frames...")
            mejorar_frames(
                directorio_temp=directorio_temp,
                ruta_exe=ruta_exe,
                ruta_modelos=ruta_modelos,
                escala=self.var_escala.get(),
                upscaler_id=upscaler_id,
                modelo_id=modelo_id,
                denoise=self.var_denoise.get(),
            )

            self._update_status("Creando video final...")
            convertir_frames_a_video(
                self.archivo_seleccionado,
                archivo_salida,
                directorio_temp
            )

            self._update_status("Mejora finalizada")

        except Exception as e:
            raise
        finally:
            shutil.rmtree(directorio_temp, ignore_errors=True)

    def _configurar_interfaz_proceso(self, habilitado: bool):
        estado = "normal" if habilitado else "disabled"
        for widget in [self.boton_mejorar, self.boton_seleccionar,
                    self.selector_upscaler, self.selector_modelo,
                    self.selector_escala, self.selector_denoise]:
            widget.configure(state=estado)
        self.barra_progreso.set(0 if habilitado else 1)
        self.label_estado.configure(text="Listo" if habilitado else "Procesando...")

    def _actualizar_interfaz(self):
        self.boton_mejorar.configure(state="normal" if self.archivo_seleccionado else "disabled")
        self.label_archivo.configure(
            text=f"Archivo: {Path(self.archivo_seleccionado).name}" if self.archivo_seleccionado
            else "Arrastra y suelta un archivo o selecciona")

    def _iniciar_proceso(self):
        ruta_exe, ruta_modelos = self._obtener_rutas_actuales()

        if not ruta_exe or not os.path.exists(ruta_exe):
            messagebox.showerror("Error", f"No se encuentra el ejecutable: {ruta_exe}")
            return

        if not ruta_modelos or not os.path.exists(ruta_modelos):
            messagebox.showerror("Error", f"No se encuentra el directorio de modelos: {ruta_modelos}")
            return

        directorio_salida = filedialog.askdirectory(title="Seleccionar ubicacion de salida")
        if not directorio_salida:
            return

        self.is_processing = True
        self._configurar_interfaz_proceso(False)

        if self.solo_imagenes.get():
            self.run_in_background(
                target=self._procesar_imagen,
                args=(directorio_salida,),
                callback=lambda *_: messagebox.showinfo("Exito", "Imagen mejorada exitosamente")
            )
        else:
            self.run_in_background(
                target=self._procesar_video,
                args=(directorio_salida,),
                callback=lambda *_: messagebox.showinfo("Exito", "Video mejorado exitosamente")
            )

    def _procesar_imagen(self, directorio_salida):
        upscaler_id = self.upscaler_var.get()
        modelo_id = self.modelo_var.get()
        ruta_exe, ruta_modelos = self._obtener_rutas_actuales()

        archivo_salida = os.path.join(
            directorio_salida,
            f"{Path(self.archivo_seleccionado).stem}_mejorado.png"
        )
        mejorar_imagen(
            self.archivo_seleccionado,
            archivo_salida,
            ruta_exe,
            ruta_modelos,
            self.var_escala.get(),
            upscaler_id=upscaler_id,
            modelo_id=modelo_id,
            denoise=self.var_denoise.get(),
        )
