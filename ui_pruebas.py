# ui_pruebas
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading
import shutil
from queue import Queue
from pathlib import Path
from prueba_config import get_realesrgan_path, get_models_path
from config.Process.video_converter import convertir_frames_a_video
from config.Process.image_processor import mejorar_imagen 
from config.Process.frame_processor import mejorar_frames, extraer_frames

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    using_dnd = True
except ImportError:
    using_dnd = False

class AplicacionMejoradorVideo(ctk.CTk if not using_dnd else TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mejorador de Video e Imagen IA Pro")
        self.geometry("1000x800")
        self._setup_inicial()
        self._crear_interfaz()
        self.queue = Queue()
        self.processing_thread = None
        self.is_processing = False
        self.frames_mejorados = []
        self.video_generado = False

    def run_in_background(self, target, args=(), callback=None):
        """Ejecuta una función en segundo plano"""
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
                callback(args) if args is not None else callback()
        except:
            pass
        finally:
            if self.is_processing:
                self.after(100, self._check_queue)

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
        # Configuración del tema
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color="#1a1a1a" if not using_dnd else None)

        # Variables de estado
        self.archivo_seleccionado = None
        self.realesrgan_path = get_realesrgan_path()
        self.models_path = get_models_path()

        if not all([self.realesrgan_path, self.models_path]):
            messagebox.showerror("Error", "No se encontró el ejecutable realesrgan-ncnn-vulkan o los modelos")
            self.destroy()
            return

        # Variables de control
        self.solo_imagenes = ctk.BooleanVar(value=False)
        self.modelo_var = ctk.StringVar(value="realesr-animevideov3")
        self.var_escala = ctk.IntVar(value=2)

        # Configuración de modelos
        self.modelos_disponibles = {
            "realesr-animevideov3": [2, 3, 4],
            "realesrgan-x4plus-anime": [4],
            #"realesrgan-x4plus": [4],
            #"ultrasharp": [4],
            #"ultramix_balanced": [4],
            #"remacri": [4],
        }

        # Configuración DnD si está disponible
        if using_dnd:
            self.drop_target_register(DND_FILES)
            self.dnd_bind('<<Drop>>', self._procesar_archivo_entrada)

    def _crear_interfaz(self):
        # Panel principal
        self.panel_principal = ctk.CTkFrame(self)
        self.panel_principal.pack(fill="both", expand=True, padx=20, pady=20)

        # Cabecera
        self._crear_cabecera()

        # Área principal
        self._crear_area_principal()

        # Panel de control
        self._crear_panel_control()

        # Barra de estado
        self._crear_barra_estado()

    def _crear_cabecera(self):
        frame_cabecera = ctk.CTkFrame(self.panel_principal)
        frame_cabecera.pack(fill="x", padx=10, pady=(0, 10))

        # Título principal
        ctk.CTkLabel(
            frame_cabecera,
            text="Mejorador de Video e Imagen IA Pro",
            font=("Roboto", 24, "bold")
        ).pack(side="left", pady=10, padx=10)

        # Botón de ayuda
        ctk.CTkButton(
            frame_cabecera,
            text="?",
            width=30,
            command=self._mostrar_ayuda
        ).pack(side="right", pady=10, padx=10)

    def _crear_area_principal(self):
        frame_principal = ctk.CTkFrame(self.panel_principal)
        frame_principal.pack(fill="both", expand=True, padx=10, pady=10)

        # Área de selección de archivo
        self._crear_selector_archivo(frame_principal)

        # Vista previa
        self._crear_vista_previa(frame_principal)

    def _crear_selector_archivo(self, parent):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=5, pady=5)
        # Etiqueta de archivo
        self.label_archivo = ctk.CTkLabel(
            frame,
            text="Arrastra y suelta un archivo o selecciona",
            font=("Roboto", 12)
        )
        self.label_archivo.pack(side="left", padx=10)
        # Frame para botones
        frame_botones = ctk.CTkFrame(frame)
        frame_botones.pack(side="right", padx=5)
        # Botón de selección
        self.boton_seleccionar = ctk.CTkButton(
            frame_botones,
            text="Seleccionar Archivo",
            command=self._seleccionar_archivo,
            width=150
        )
        self.boton_seleccionar.pack(side="left", padx=5)
        # Botón de limpiar
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
        # Etiqueta de vista previa
        ctk.CTkLabel(
            frame,
            text="Vista Previa",
            font=("Roboto", 14, "bold")
        ).pack(pady=5)
        # Canvas para vista previa
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
        # Frame izquierdo para opciones básicas
        frame_izq = ctk.CTkFrame(frame)
        frame_izq.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        # Checkbox para modo imagen
        ctk.CTkCheckBox(
            frame_izq,
            text="Solo procesar imágenes",
            variable=self.solo_imagenes,
            command=self._actualizar_interfaz
        ).pack(side="left", padx=10)
        # Frame derecho para controles avanzados
        frame_der = ctk.CTkFrame(frame)
        frame_der.pack(side="right", fill="x", expand=True, padx=5, pady=5)
        # Selector de modelo
        ctk.CTkLabel(frame_der, text="Modelo:").pack(side="left", padx=5)
        self.selector_modelo = ctk.CTkOptionMenu(
            frame_der,
            values=list(self.modelos_disponibles.keys()),
            variable=self.modelo_var,
            command=self._actualizar_escalas,
            width=200
        )
        self.selector_modelo.pack(side="left", padx=5)
        # Selector de escala
        ctk.CTkLabel(frame_der, text="Escala:").pack(side="left", padx=5)
        self.selector_escala = ctk.CTkOptionMenu(
            frame_der,
            values=["2x", "3x", "4x"],
            command=lambda x: self.var_escala.set(int(x[0])),
            width=100
        )
        self.selector_escala.pack(side="left", padx=5)
        # Botón de proceso
        self.boton_mejorar = ctk.CTkButton(
            frame_der,
            text="Iniciar Mejora",
            command=self._iniciar_proceso,
            state="disabled",
            width=150
        )
        self.boton_mejorar.pack(side="right", padx=10)

    def _crear_barra_estado(self):
        frame = ctk.CTkFrame(self.panel_principal)
        frame.pack(fill="x", padx=10, pady=5)
        # Frame para información del proceso
        frame_info = ctk.CTkFrame(frame)
        frame_info.pack(fill="x", padx=5, pady=5)
        # Barra de progreso
        self.barra_progreso = ctk.CTkProgressBar(frame_info)
        self.barra_progreso.pack(fill="x", padx=5, pady=5)
        self.barra_progreso.set(0)
        # Estado del proceso
        self.label_estado = ctk.CTkLabel(
            frame_info,
            text="Listo",
            font=("Roboto", 12)
        )
        self.label_estado.pack(pady=5)
        # Información adicional
        self.label_info = ctk.CTkLabel(
            frame_info,
            text="",
            font=("Roboto", 10)
        )
        self.label_info.pack(pady=2)

    def _mostrar_ayuda(self):
        messagebox.showinfo("Ayuda",
            "Instrucciones de uso:\n\n"
            "1. Selecciona un archivo de video o imagen\n"
            "2. Elige el modelo de mejora deseado\n"
            "3. Ajusta la escala de mejora\n"
            "4. Configura la calidad de salida\n"
            "5. Presiona 'Iniciar Mejora'\n\n"
            "Formatos soportados:\n"
            "- Videos: MP4, AVI, MKV, MOV\n"
            "- Imágenes: PNG, JPG, JPEG, BMP, WEBP"
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
            # Mostrar información del archivo
            self._mostrar_info_archivo()

    def _validar_archivo(self, archivo):
        ext = Path(archivo).suffix.lower()
        if self.solo_imagenes.get():
            return ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
        return ext in ['.mp4', '.avi', '.mkv', '.mov']

    def _mostrar_info_archivo(self):
        if not self.archivo_seleccionado:
            return
        path = Path(self.archivo_seleccionado)
        size = os.path.getsize(self.archivo_seleccionado) / (1024 * 1024)  # MB
        if self.solo_imagenes.get():
            img = Image.open(self.archivo_seleccionado)
            info = f"Tamaño: {size:.1f}MB | Dimensiones: {img.size[0]}x{img.size[1]}"
        else:
            cap = cv2.VideoCapture(self.archivo_seleccionado)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frames / fps if fps else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            info = f"Tamaño: {size:.1f}MB | Dimensiones: {width}x{height} | FPS: {fps:.1f} | Duración: {duration:.1f}s"
        self.label_info.configure(text=info)

    def _mostrar_vista_previa(self):
        try:
            if self.solo_imagenes.get():
                imagen = Image.open(self.archivo_seleccionado)
            else:
                cap = cv2.VideoCapture(self.archivo_seleccionado)
                ret, frame = cap.read()
                if not ret:
                    raise Exception("No se pudo leer el archivo de video")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                imagen = Image.fromarray(frame)
                cap.release()
            # Ajustar tamaño y mostrar
            w, h = imagen.size
            ratio = min(800/w, 400/h)
            imagen = imagen.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
            foto = ImageTk.PhotoImage(imagen)
            self.canvas_preview.delete("all")
            self.canvas_preview.create_image(400, 200, image=foto)
            self.canvas_preview.image = foto
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar la vista previa: {str(e)}")

    def _procesar_video(self, directorio_salida):
        directorio_temp = os.path.join(directorio_salida, "temp_frames")
        os.makedirs(directorio_temp, exist_ok=True)
        self.archivo_salida = os.path.join(directorio_salida,
                                    f"{Path(self.archivo_seleccionado).stem}_mejorado.mp4")
        try:
            # Proceso secuencial de video
            self.label_estado.configure(text="Extrayendo frames...")
            extraer_frames(self.archivo_seleccionado, directorio_temp,
                        self.barra_progreso, self.after)
            
            self.label_estado.configure(text="Mejorando frames...")
            mejorar_frames(
                directorio_temp=directorio_temp,
                ruta_realesrgan=self.realesrgan_path,
                ruta_modelos=self.models_path,
                barra_progreso=self.barra_progreso,
                escala=self.var_escala.get(),
                modelo=self.modelo_var.get()
            )
            
            self.label_estado.configure(text="Creando video final...")
            convertir_frames_a_video(
                self.archivo_seleccionado, 
                self.archivo_salida,
                directorio_temp, 
                self.barra_progreso, 
                self.finalizar_mejora
            )
        except Exception as e:
            self._handle_error(str(e))
        finally:
            shutil.rmtree(directorio_temp, ignore_errors=True)
            
    def _configurar_interfaz_proceso(self, habilitado: bool):
        estado = "normal" if habilitado else "disabled"
        for widget in [self.boton_mejorar, self.boton_seleccionar,
                    self.selector_modelo, self.selector_escala]:
            widget.configure(state=estado)
        self.barra_progreso.set(0 if habilitado else 1)
        self.label_estado.configure(text="Listo" if habilitado else "Procesando...")

    def _actualizar_escalas(self, *args):
        modelo = self.modelo_var.get()
        escalas = self.modelos_disponibles.get(modelo, [2])
        self.selector_escala.configure(values=[f"{x}x" for x in escalas])
        self.selector_escala.set(f"{escalas[0]}x")
        self.var_escala.set(escalas[0])

    def _actualizar_interfaz(self):
        self.boton_mejorar.configure(state="normal" if self.archivo_seleccionado else "disabled")
        self.label_archivo.configure(
            text=f"Archivo: {Path(self.archivo_seleccionado).name}" if self.archivo_seleccionado
            else "Arrastra y suelta un archivo o selecciona")

    def _iniciar_proceso(self):
        # Asegurarse de que las rutas sean correctas
        modelo_path = os.path.abspath(self.realesrgan_path)
        models_path = os.path.abspath(self.models_path)

        # Verificar que los archivos existen
        if not os.path.exists(modelo_path):
            messagebox.showerror("Error", f"No se encuentra el ejecutable en: {modelo_path}")
            return

        if not os.path.exists(models_path):
            messagebox.showerror("Error", f"No se encuentra el directorio de modelos en: {models_path}")
            return
        directorio_salida = filedialog.askdirectory(title="Seleccionar ubicación de salida")
        if not directorio_salida:
            return
        self.is_processing = True
        self._configurar_interfaz_proceso(False)

        if self.solo_imagenes.get():
            self.run_in_background(
                target=self._procesar_imagen,
                args=(directorio_salida,),
                callback=lambda _: messagebox.showinfo("Éxito", "Proceso completado exitosamente")
            )
        else:
            self.run_in_background(
                target=self._procesar_video,
                args=(directorio_salida,),
                callback=lambda _: messagebox.showinfo("Éxito", "Proceso completado exitosamente")
            )
            
    def _procesar_imagen(self, directorio_salida):
        archivo_salida = os.path.join(directorio_salida, f"{Path(self.archivo_seleccionado).stem}_mejorado.png")
        try:
            mejorar_imagen(
                self.archivo_seleccionado, 
                archivo_salida, 
                self.realesrgan_path, 
                self.models_path, 
                self.var_escala.get(), 
                self.modelo_var.get()
            )
            messagebox.showinfo("Éxito", f"Imagen mejorada guardada en:\n{archivo_salida}")
        except Exception as e:
            self._handle_error(str(e))
            
    def iniciar_mejora(self):
        try:
            # Lógica para procesar y mejorar imágenes
            self.procesar_frames()
        except Exception as e:
            print(f"Error durante la mejora: {e}")
        finally:
            # Llamar al método 'finalizar_mejora'
            self.finalizar_mejora()

    def procesar_frames(self):
        # Aquí se realiza el procesamiento de los frames
        print("Procesando frames...")
        # Simular que el procesamiento ocurre correctamente
        self.frames_mejorados = ["frame1.png", "frame2.png"]  # Ejemplo

        # Generar video después de procesar
        self.generar_video_mejorado()

    def generar_video_mejorado(self):
        try:
            if not self.frames_mejorados:
                raise ValueError("No hay frames mejorados para generar el video.")

            # Lógica para combinar los frames en un video
            print("Generando video mejorado...")
            self.video_generado = True
        except Exception as e:
            print(f"Error al generar el video: {e}")

    def finalizar_mejora(self):
        # Método que se llama al finalizar todo el proceso
        ruta_video = getattr(self, 'archivo_salida', 'Video no generado')
        
        mensaje_exito = f"""
    Proceso de Mejora Completado ✅
    --------------------------------
    Archivo de salida: {ruta_video}
    Modelo utilizado: {self.modelo_var.get()}
    Escala de mejora: {self.var_escala.get()}x
        """
        
        # Actualizar label de estado con mensaje de éxito
        self.label_estado.configure(text="Mejora finalizada con éxito")
        
        # Opcional:def _end_processing(self):
        """Finaliza el procesamiento"""
        self.is_processing = False
        self._configurar_interfaz_proceso(True)
