from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import uvicorn
from PyQt6.Qsci import QsciLexerPython, QsciScintilla
from PyQt6.QtCore import QProcess, Qt, QUrl
from PyQt6.QtGui import (
    QAction,
    QColor,
    QFileSystemModel,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QKeySequence,
)
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTabWidget,
    QToolBar,
    QTreeView,
)

from axiom import __version__
from axiom.cognitive_agent import CognitiveAgent
from axiom.config import (
    DEFAULT_BRAIN_FILE,
    DEFAULT_STATE_FILE,
    PROJECT_ROOT,
    STATIC_DIR,
)
from axiom.scripts.studio_server import app as fastapi_app

server_started_event = threading.Event()
server_shutdown_event = threading.Event()
training_thread: threading.Thread | None = None
stop_event = threading.Event()


class AxiomWebPage(QWebEnginePage):
    """Custom WebEnginePage to intercept axiom:// scheme."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

    def acceptNavigationRequest(self, url, _type, isMainFrame):  # noqa: N802, N803
        if url.scheme() == "axiom":
            host = url.host()
            if host == "browse-model":
                if self.main_window:
                    self.main_window.browse_model_dialog()
                return False
            if host == "open-folder":
                if self.main_window:
                    self.main_window.open_folder_dialog()
                return False
            if host == "open-recent":
                # Extract path from query params
                from PyQt6.QtCore import QUrlQuery

                query = QUrlQuery(url)
                path = query.queryItemValue("path")
                if path and self.main_window:
                    self.main_window.open_workspace(path)
                return False
            if host == "open-agent-manager":
                if self.main_window:
                    self.main_window.open_agent_manager_dialog()
                return False
            if host == "clone-repo":
                if self.main_window:
                    self.main_window.open_clone_repo_dialog()
                return False

        return super().acceptNavigationRequest(url, _type, isMainFrame)


def run_server():
    """
    This function will be run in a background thread to start the uvicorn server.
    """
    config = uvicorn.Config(
        app=fastapi_app, host="127.0.0.1", port=8000, log_level="info"
    )
    server = uvicorn.Server(config)

    def server_runner():
        server.run()
        server_shutdown_event.set()

    server_thread = threading.Thread(target=server_runner)
    server_thread.start()

    while not server.started:
        time.sleep(0.01)

    server_started_event.set()
    server_shutdown_event.wait()
    if hasattr(server, "should_exit"):
        server.should_exit = True
    server_thread.join()


class FileBrowser(QTreeView):
    """
    Custom QTreeView to handle drag-and-drop and context menus.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QTreeView.DragDropMode.InternalMove)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _show_context_menu(self, position):
        index = self.indexAt(position)
        if not index.isValid():
            return

        # Actually, we should create a QMenu directly
        from PyQt6.QtWidgets import QMenu

        menu = QMenu(self)

        copy_path_action = QAction("Copy Path", self)
        copy_path_action.triggered.connect(lambda: self._copy_path(index))
        menu.addAction(copy_path_action)

        copy_rel_path_action = QAction("Copy Relative Path", self)
        copy_rel_path_action.triggered.connect(lambda: self._copy_rel_path(index))
        menu.addAction(copy_rel_path_action)

        menu.addSeparator()

        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._delete_file(index))
        menu.addAction(delete_action)

        menu.exec(self.viewport().mapToGlobal(position))

    def _copy_path(self, index):
        path = self.model().filePath(index)
        QApplication.clipboard().setText(path)

    def _copy_rel_path(self, index):
        path = Path(self.model().filePath(index))
        try:
            rel_path = path.relative_to(PROJECT_ROOT)
            QApplication.clipboard().setText(str(rel_path))
        except ValueError:
            QApplication.clipboard().setText(str(path))

    def _delete_file(self, index):
        from PyQt6.QtWidgets import QMessageBox

        path = Path(self.model().filePath(index))
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete {path.name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if path.is_dir():
                    import shutil

                    shutil.rmtree(path)
                else:
                    path.unlink()
            except Exception as e:
                print(f"Error deleting {path}: {e}")


class AxiomStudio(QMainWindow):
    """
    The main window for the Axiom Studio IDE, now with a full IDE layout.
    """

    def __init__(self, agent: CognitiveAgent):
        super().__init__()
        self.agent = agent
        self.setWindowTitle(f"Axiom Studio v{__version__}")
        self.setGeometry(100, 100, 1600, 1000)
        self.current_file_path: Path | None = None

        # Main Splitter (Horizontal): [File Browser] | [Right Area]
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)

        # 1. File Browser (Left)
        self.file_browser = FileBrowser(self)
        self.file_system_model = QFileSystemModel()
        self.file_system_model.setRootPath(str(PROJECT_ROOT))
        self.file_system_model.setReadOnly(
            False
        )  # Enable write operations for drag/drop
        self.file_browser.setModel(self.file_system_model)
        self.file_browser.setRootIndex(self.file_system_model.index(str(PROJECT_ROOT)))
        self.file_browser.setColumnWidth(0, 250)
        self.file_browser.doubleClicked.connect(self.open_file_in_editor)
        main_splitter.addWidget(self.file_browser)

        # Right Area Splitter (Horizontal): [Editor/Tabs] | [AI Chat]
        self.right_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(self.right_splitter)

        # 2. Central Area (Vertical Splitter: Workspace / Bottom Panel)
        self.center_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.addWidget(self.center_splitter)

        # 2a. Workspace (Top)
        self.workspace = QTabWidget()
        self.center_splitter.addWidget(self.workspace)

        self.browser = QWebEngineView()
        # Use custom page to handle axiom:// links
        self.browser_page = AxiomWebPage(self)
        self.browser.setPage(self.browser_page)

        # Start at Welcome Screen
        url = QUrl("http://127.0.0.1:8000/welcome")
        self.browser.setUrl(url)
        self.workspace.addTab(self.browser, "Agent Console")

        self.editor = QsciScintilla()
        self._configure_editor()
        self.workspace.addTab(self.editor, "Code Editor")

        # Connect tab change AFTER editor is created
        self.workspace.currentChanged.connect(self.on_tab_changed)

        # 2b. Bottom Panel (Terminal)
        self.bottom_panel = QTabWidget()
        self.bottom_panel.setTabPosition(
            QTabWidget.TabPosition.South
        )  # Tabs at bottom like VS Code
        self.center_splitter.addWidget(self.bottom_panel)

        # Initial Terminal
        self.create_new_terminal()
        # self.create_new_terminal() # This function is not yet defined, commenting out for now.

        # Set initial sizes (70% editor, 30% terminal)
        self.center_splitter.setSizes([700, 300])

        # 3. AI Chat Panel (Right)
        self.chat_view = QWebEngineView()
        self.chat_page = AxiomWebPage(self)
        self.chat_view.setPage(self.chat_page)
        self.chat_view.setUrl(QUrl("http://127.0.0.1:8000/chat"))
        self.chat_view.setMinimumWidth(300)
        main_splitter.addWidget(self.chat_view)

        # Set initial splitter sizes (20% file browser, 60% editor, 20% chat)
        main_splitter.setSizes([300, 900, 400])  # Browser vs Rest
        # The right_splitter now only contains the center_splitter, so this line is redundant/incorrect for the new structure.
        # self.right_splitter.setSizes([950, 400])  # Workspace vs Chat

        self._create_menubar()
        self._create_toolbar()

        # --- Fresh Start Layout: Hide Panels Initially ---
        self.file_browser.hide()
        self.bottom_panel.hide()
        self.chat_view.hide()
        # Ensure workspace (welcome screen) is visible
        self.workspace.show()

    def open_agent_manager_dialog(self):
        """Opens the Agent Manager in a new window."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Agent Manager")
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        view = QWebEngineView()
        page = AxiomWebPage(self)
        view.setPage(page)
        view.setUrl(QUrl("http://localhost:8000/agent_manager"))
        layout.addWidget(view)

        dialog.exec()

    def open_clone_repo_dialog(self):
        """Opens the Git Clone tool in a new window."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Clone Repository")
        dialog.resize(600, 400)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        view = QWebEngineView()
        page = AxiomWebPage(self)
        view.setPage(page)
        view.setUrl(QUrl("http://localhost:8000/clone"))
        layout.addWidget(view)

        dialog.exec()

    def _create_menubar(self):
        """Creates the standard IDE menu bar."""
        menubar = self.menuBar()

        # --- File Menu ---
        file_menu = menubar.addMenu("&File")

        open_folder_action = QAction("Open Folder...", self)
        open_folder_action.setShortcut(QKeySequence.StandardKey.Open)
        open_folder_action.triggered.connect(self.open_folder_dialog)
        file_menu.addAction(open_folder_action)

        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- Edit Menu ---
        edit_menu = menubar.addMenu("&Edit")

        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.editor.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self.editor.redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        cut_action = QAction("Cut", self)
        cut_action.setShortcut(QKeySequence.StandardKey.Cut)
        cut_action.triggered.connect(self.editor.cut)
        edit_menu.addAction(cut_action)

        copy_action = QAction("Copy", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self.editor.copy)
        edit_menu.addAction(copy_action)

        paste_action = QAction("Paste", self)
        paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        paste_action.triggered.connect(self.editor.paste)
        edit_menu.addAction(paste_action)

        # --- Agent Menu ---
        agent_menu = menubar.addMenu("&Agent")

        agent_manager_action = QAction("Agent Manager", self)
        agent_manager_action.triggered.connect(self.open_agent_manager_dialog)
        agent_menu.addAction(agent_manager_action)

        # --- View Menu ---
        view_menu = menubar.addMenu("&View")

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(self.editor.zoomIn)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(self.editor.zoomOut)
        view_menu.addAction(zoom_out_action)

        view_menu.addSeparator()

        # --- Run Menu ---
        run_menu = menubar.addMenu("Run")

        run_action = QAction("Run Agent", self)
        run_action.triggered.connect(self.start_training_clicked)
        run_menu.addAction(run_action)

        stop_action = QAction("Stop Agent", self)
        stop_action.triggered.connect(self.stop_training_clicked)
        run_menu.addAction(stop_action)

        # --- Terminal Menu ---
        terminal_menu = menubar.addMenu("Terminal")

        new_term_action = QAction("New Terminal", self)
        new_term_action.setShortcut(QKeySequence("Ctrl+Shift+`"))
        new_term_action.triggered.connect(self.create_new_terminal)
        terminal_menu.addAction(new_term_action)

        toggle_term_action = QAction("Toggle Terminal", self)
        toggle_term_action.setShortcut(QKeySequence("Ctrl+`"))
        toggle_term_action.triggered.connect(self.toggle_bottom_panel)
        terminal_menu.addAction(toggle_term_action)

        # --- Window Menu ---
        window_menu = menubar.addMenu("Window")
        toggle_chat_action = QAction("Toggle AI Chat", self)
        toggle_chat_action.setShortcut("Ctrl+Shift+L")  # Common shortcut for sidebars
        toggle_chat_action.triggered.connect(self.toggle_chat_panel)
        window_menu.addAction(toggle_chat_action)

    def _create_toolbar(self):
        """Creates the native toolbar with lifecycle controls."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Metallic Button Style
        metallic_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4b5563, stop:1 #374151);
                color: white;
                border: 1px solid #1f2937;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6b7280, stop:1 #4b5563);
            }
            QPushButton:pressed {
                background: #1f2937;
            }
            QPushButton:disabled {
                background: #374151;
                color: #9ca3af;
            }
        """

        # --- Play Button ---
        self.play_button = QPushButton("▶ Start")
        self.play_button.setStyleSheet(metallic_style)
        self.play_button.clicked.connect(self.start_training_clicked)
        toolbar.addWidget(self.play_button)

        # --- Stop Button ---
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setStyleSheet(metallic_style)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training_clicked)
        toolbar.addWidget(self.stop_button)

        toolbar.addSeparator()

        # --- Open Folder ---
        self.open_folder_button = QPushButton("📂 Open")
        self.open_folder_button.setStyleSheet(metallic_style)
        self.open_folder_button.clicked.connect(self.open_folder_dialog)
        toolbar.addWidget(self.open_folder_button)

        toolbar.addSeparator()

        # --- Panel Toggles ---
        # Toggle Button Style (slightly different to show state)
        toggle_style = """
            QPushButton {
                background: transparent;
                color: #d1d5db;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background: #374151;
            }
            QPushButton:checked {
                background: #4b5563;
                color: white;
                border: 1px solid #6b7280;
            }
        """

        self.toggle_browser_btn = QPushButton("Files")
        self.toggle_browser_btn.setCheckable(True)
        self.toggle_browser_btn.setChecked(False)
        self.toggle_browser_btn.setToolTip("Toggle File Explorer")
        self.toggle_browser_btn.clicked.connect(self.toggle_file_browser)
        self.toggle_browser_btn.setStyleSheet(toggle_style)
        toolbar.addWidget(self.toggle_browser_btn)

        self.toggle_term_btn = QPushButton("Terminal")
        self.toggle_term_btn.setCheckable(True)
        self.toggle_term_btn.setChecked(False)
        self.toggle_term_btn.setToolTip("Toggle Terminal Panel")
        self.toggle_term_btn.clicked.connect(self.toggle_bottom_panel)
        self.toggle_term_btn.setStyleSheet(toggle_style)
        toolbar.addWidget(self.toggle_term_btn)

        self.toggle_chat_btn = QPushButton("Chat")
        self.toggle_chat_btn.setCheckable(True)
        self.toggle_chat_btn.setChecked(False)
        self.toggle_chat_btn.setToolTip("Toggle AI Chat Panel")
        self.toggle_chat_btn.clicked.connect(self.toggle_chat_panel)
        self.toggle_chat_btn.setStyleSheet(toggle_style)
        toolbar.addWidget(self.toggle_chat_btn)

        toolbar.addSeparator()

        # --- Agent Manager ---
        self.agent_manager_btn = QPushButton("⚙ Manager")
        self.agent_manager_btn.setToolTip("Open Agent Manager Settings")
        self.agent_manager_btn.clicked.connect(self.open_agent_manager_dialog)
        self.agent_manager_btn.setStyleSheet(metallic_style)
        toolbar.addWidget(self.agent_manager_btn)

    def _configure_editor(self):
        """Sets up the QScintilla code editor with a VS Code Dark+ inspired theme."""
        # --- Colors (VS Code Dark+ Palette) ---
        color_background = QColor("#1e1e1e")
        color_foreground = QColor("#d4d4d4")
        color_caret = QColor("#aeafad")
        color_selection = QColor("#264f78")
        color_margin_bg = QColor("#1e1e1e")
        color_margin_text = QColor("#858585")

        # Syntax Colors
        color_comment = QColor("#6a9955")
        color_string = QColor("#ce9178")
        color_number = QColor("#b5cea8")
        color_keyword = QColor("#569cd6")
        color_class = QColor("#4ec9b0")
        color_function = QColor("#dcdcaa")
        color_decorator = QColor("#dcdcaa")
        color_operator = QColor("#d4d4d4")
        color_identifier = QColor("#d4d4d4")

        # --- Font Setup ---
        # Try to load bundled JetBrains Mono
        font_path = STATIC_DIR / "fonts" / "JetBrainsMono-Regular.ttf"
        font_id = QFontDatabase.addApplicationFont(str(font_path))

        if font_id != -1:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
            print(f"--- [Axiom Studio]: Loaded bundled font: {font_family} ---")
        else:
            # Fallback to system monospaced fonts
            font_family = "Menlo" if sys.platform == "darwin" else "Consolas"
            print(
                f"--- [Axiom Studio]: Bundled font not found. Using fallback: {font_family} ---"
            )

        editor_font = QFont(font_family)
        editor_font.setPointSize(13)
        editor_font.setFixedPitch(True)
        editor_font.setStyleHint(QFont.StyleHint.Monospace)

        # Verify it's actually fixed pitch
        fm = QFontMetrics(editor_font)
        if not fm.horizontalAdvance("i") == fm.horizontalAdvance("W"):
            print(
                "--- [Axiom Studio]: WARNING: Font is not fixed pitch! Text may overlap. ---"
            )

        # --- General Editor Settings ---
        self.editor.setFont(editor_font)
        self.editor.setUtf8(True)
        self.editor.setPaper(color_background)
        self.editor.setColor(color_foreground)
        self.editor.setCaretForegroundColor(color_caret)
        self.editor.setSelectionBackgroundColor(color_selection)
        self.editor.setSelectionForegroundColor(color_foreground)

        # Margins (Line Numbers)
        font_metrics = self.fontMetrics()
        self.editor.setMarginWidth(0, font_metrics.horizontalAdvance("00000") + 5)
        self.editor.setMarginLineNumbers(0, True)
        self.editor.setMarginsBackgroundColor(color_margin_bg)
        self.editor.setMarginsForegroundColor(color_margin_text)
        self.editor.setMarginsFont(editor_font)

        # Indentation
        self.editor.setIndentationsUseTabs(False)
        self.editor.setTabWidth(4)
        self.editor.setIndentationGuides(True)
        self.editor.setIndentationGuidesBackgroundColor(color_selection)
        self.editor.setAutoIndent(True)

        # Brace Matching
        self.editor.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
        self.editor.setMatchedBraceBackgroundColor(color_selection)
        self.editor.setMatchedBraceForegroundColor(QColor("#ffffff"))

        # Current Line Highlight
        self.editor.setCaretLineVisible(True)
        self.editor.setCaretLineBackgroundColor(QColor("#282828"))

        # --- Lexer (Python) Configuration ---
        lexer = QsciLexerPython()
        lexer.setDefaultFont(editor_font)  # CRITICAL: Ensure lexer uses same font
        lexer.setDefaultPaper(color_background)
        lexer.setDefaultColor(color_foreground)

        # Apply Colors AND FONT to Lexer Styles
        # We must explicitly set the font for every style to prevent overlapping
        for style in range(33):  # Python lexer has ~33 styles
            lexer.setColor(color_foreground, style)
            lexer.setPaper(color_background, style)
            lexer.setFont(editor_font, style)

        # Specific Style Overrides
        lexer.setColor(color_comment, 1)  # Comment
        lexer.setColor(color_number, 2)  # Number
        lexer.setColor(color_string, 3)  # Double-quoted string
        lexer.setColor(color_string, 4)  # Single-quoted string
        lexer.setColor(color_keyword, 5)  # Keyword
        lexer.setColor(color_string, 6)  # Triple-quoted string
        lexer.setColor(color_string, 7)  # Triple-single-quoted string
        lexer.setColor(color_class, 8)  # Class name
        lexer.setColor(color_function, 9)  # Function name
        lexer.setColor(color_operator, 10)  # Operator
        lexer.setColor(color_identifier, 11)  # Identifier
        lexer.setColor(color_comment, 12)  # Comment block
        lexer.setColor(QColor("#f44747"), 13)  # Unclosed string
        lexer.setColor(color_identifier, 14)  # Highlighted identifier
        lexer.setColor(color_decorator, 15)  # Decorator

        self.editor.setLexer(lexer)

    def open_file_in_editor(self, index):
        """Slot to handle double-clicking a file in the QTreeView."""
        path = self.file_system_model.filePath(index)
        file_path = Path(path)
        if file_path.is_file():  # Allow opening any file, not just .py
            try:
                content = file_path.read_text(encoding="utf-8")
                self.editor.setText(content)
            except Exception as e:
                print(f"--- [Axiom Studio]: Error opening file {path}: {e} ---")
                return  # Exit if file cannot be read

            self.current_file_path = file_path
            self.setWindowTitle(f"Axiom Studio - {file_path.name}")

            # Switch to editor tab
            self.workspace.setCurrentWidget(self.editor)
            self.update_chat_context()  # Force update

    def save_file(self):
        """Saves the current file in the editor."""
        if self.current_file_path:
            try:
                content = self.editor.text()
                self.current_file_path.write_text(content, encoding="utf-8")
                print(f"--- [Axiom Studio]: Saved {self.current_file_path} ---")
                # Optional: Visual feedback like flashing the status bar
            except Exception as e:
                print(f"--- [Axiom Studio]: Error saving file: {e} ---")
        else:
            # If no file is open (e.g. new buffer), we could ask for a path
            # For now, just print a warning
            print("--- [Axiom Studio]: No file currently open to save. ---")

    def handle_process_state_change(self, new_state):
        """Re-enables the Play button when a process finishes."""
        if new_state == QProcess.ProcessState.NotRunning:
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def open_folder_dialog(self):
        """Opens a dialog to select a workspace folder."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Workspace Folder", str(PROJECT_ROOT)
        )
        if folder_path:
            self.open_workspace(folder_path)

    def open_workspace(self, folder_path: str):
        """Switches the workspace to the given folder and updates history."""
        # Set the root path for the file system model
        self.file_system_model.setRootPath(folder_path)

        # Set the file browser to show only this workspace
        root_index = self.file_system_model.index(folder_path)
        self.file_browser.setRootIndex(root_index)

        # Expand the root to show immediate children
        self.file_browser.expand(root_index)

        print(f"--- [Axiom Studio]: Workspace changed to {folder_path} ---")

        # Update history
        self._add_recent_workspace(folder_path)

        # Reveal Panels
        # Reveal Panels
        self.file_browser.show()
        if hasattr(self, "toggle_browser_btn"):
            self.toggle_browser_btn.setChecked(True)

        self.bottom_panel.show()
        if hasattr(self, "toggle_term_btn"):
            self.toggle_term_btn.setChecked(True)

        self.chat_view.show()
        if hasattr(self, "toggle_chat_btn"):
            self.toggle_chat_btn.setChecked(True)

        # Navigate browser to Dashboard if on Welcome screen
        current_url = self.browser.url().toString()
        if "welcome" in current_url:
            self.browser.setUrl(QUrl("http://127.0.0.1:8000/studio"))

    def _add_recent_workspace(self, path: str):
        """Adds a path to the recent workspaces list in ~/.axiom/studio_config.json."""
        from urllib.parse import unquote

        # Normalize path: decode URL chars, resolve absolute path, remove trailing slash
        path = unquote(path)
        path = str(Path(path).resolve())

        config_dir = Path.home() / ".axiom"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "studio_config.json"

        data = {}
        if config_file.exists():
            try:
                data = json.loads(config_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        recents = data.get("recent_workspaces", [])
        # Remove if exists (to move to top)
        if path in recents:
            recents.remove(path)

        recents.insert(0, path)
        # Keep max 10
        recents = recents[:10]

        data["recent_workspaces"] = recents
        config_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def toggle_chat_panel(self):
        """Toggles the visibility of the AI Chat Panel."""
        if self.chat_view.isVisible():
            self.chat_view.hide()
            if hasattr(self, "toggle_chat_btn"):
                self.toggle_chat_btn.setChecked(False)
        else:
            self.chat_view.show()
            if hasattr(self, "toggle_chat_btn"):
                self.toggle_chat_btn.setChecked(True)

    def toggle_file_browser(self):
        """Toggles the visibility of the File Browser."""
        if self.file_browser.isVisible():
            self.file_browser.hide()
            if hasattr(self, "toggle_browser_btn"):
                self.toggle_browser_btn.setChecked(False)
        else:
            self.file_browser.show()
            if hasattr(self, "toggle_browser_btn"):
                self.toggle_browser_btn.setChecked(True)

    def on_tab_changed(self, index):
        """Handle tab changes to update context."""
        current_widget = self.workspace.widget(index)
        if current_widget == self.editor:
            self.update_chat_context()
        else:
            # Clear context if not in editor
            self.chat_view.page().runJavaScript("window.setActiveFile(null, null);")

    def browse_model_dialog(self):
        """Opens a file dialog to select a GGUF model."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GGUF Model",
            str(Path.home()),
            "GGUF Models (*.gguf);;All Files (*)",
        )

        if file_path:
            # Reload the model in the agent
            from axiom.scripts.studio_server import agent

            if agent and hasattr(agent, "interpreter"):
                success = agent.interpreter.reload_model(file_path)
                if success:
                    # Update UI
                    model_name = Path(file_path).name
                    js_code = f"addModelOption('{model_name}', '{file_path}')"
                    self.chat_view.page().runJavaScript(js_code)
                    self.chat_view.page().runJavaScript(
                        "alert('Model reloaded successfully!');"
                    )
                else:
                    self.chat_view.page().runJavaScript(
                        "alert('Failed to reload model. Check logs.');"
                    )
            else:
                self.chat_view.page().runJavaScript(
                    "alert('Agent not initialized. Start the agent first.');"
                )

    def update_chat_context(self):
        """Injects the current editor content into the chat view."""
        if not self.current_file_path:
            return

        file_name = self.current_file_path.name
        content = self.editor.text()

        # Escape content for JS string
        content_js = json.dumps(content)  # Handles escaping quotes, newlines, etc.
        file_name_js = json.dumps(file_name)

        js_code = f"window.setActiveFile({file_name_js}, {content_js});"
        self.chat_view.page().runJavaScript(js_code)

    def create_new_terminal(self):
        """Creates a new terminal tab in the bottom panel."""
        term_view = QWebEngineView()
        term_view.setUrl(QUrl("http://127.0.0.1:8000/console"))

        count = self.bottom_panel.count() + 1
        self.bottom_panel.addTab(term_view, f"Terminal {count}")
        self.bottom_panel.setCurrentWidget(term_view)

        # Ensure panel is visible
        if not self.bottom_panel.isVisible():
            self.bottom_panel.show()
            if hasattr(self, "toggle_term_btn"):
                self.toggle_term_btn.setChecked(True)

    def toggle_bottom_panel(self):
        """Toggles the visibility of the bottom terminal panel."""
        if self.bottom_panel.isVisible():
            self.bottom_panel.hide()
            if hasattr(self, "toggle_term_btn"):
                self.toggle_term_btn.setChecked(False)
        else:
            self.bottom_panel.show()
            if hasattr(self, "toggle_term_btn"):
                self.toggle_term_btn.setChecked(True)

    def start_training_clicked(self):
        """
        Handles the 'Play Start' button click by injecting the 'axiom-train'
        command into the terminal web view.
        """
        # Ensure we have a terminal
        if self.bottom_panel.count() == 0:
            self.create_new_terminal()

        # Show panel
        if not self.bottom_panel.isVisible():
            self.bottom_panel.show()
            if hasattr(self, "toggle_term_btn"):
                self.toggle_term_btn.setChecked(True)

        # Get current terminal view
        current_term = self.bottom_panel.currentWidget()
        if isinstance(current_term, QWebEngineView):
            # The '\n' is the equivalent of pressing Enter
            # User requested specific debug command sequence
            cmd = "cd '/Volumes/SSD 1/REPO/Axiom-Agent' && source .venv/bin/activate && export AXIOM_LOG_LEVEL=DEBUG && AXIOM_LOG_LEVEL=DEBUG axiom-train"
            js_command = f'window.sendToTerminal("{cmd}\\n");'
            current_term.page().runJavaScript(js_command)

        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_training_clicked(self):
        """
        Handles the 'Stop' button click by injecting the Ctrl+C character
        into the terminal web view.
        """
        # Ensure panel is visible
        if not self.bottom_panel.isVisible():
            self.bottom_panel.show()
            if hasattr(self, "toggle_term_btn"):
                self.toggle_term_btn.setChecked(True)

        current_term = self.bottom_panel.currentWidget()
        if isinstance(current_term, QWebEngineView):
            # The character \x03 is the ASCII code for Ctrl+C
            js_command = "window.sendToTerminal('\\x03');"
            current_term.page().runJavaScript(js_command)

        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def close_event(self, event):
        """Overrides the default close event to gracefully shut down the server."""
        print(
            "--- [Axiom Studio]: Close button clicked. Shutting down backend server... ---"
        )
        server_shutdown_event.set()
        event.accept()


def main() -> int:
    """
    The main entry point for the native Axiom Studio application.
    """
    # We still need an agent for the interactive chat in the web view
    print("--- [Axiom Studio]: Initializing interactive CognitiveAgent... ---")
    agent = CognitiveAgent(
        brain_file=DEFAULT_BRAIN_FILE,
        state_file=DEFAULT_STATE_FILE,
        inference_mode=False,
    )
    agent.load_state()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    print("--- [Axiom Studio]: Waiting for backend server to start... ---")
    server_started_event.wait()
    print("--- [Axiom Studio]: Backend server started. Launching UI. ---")

    app = QApplication(sys.argv)

    font_path = STATIC_DIR / "fonts" / "JetBrainsMono-Regular.ttf"
    if font_path.exists():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id == -1:
            print(
                "--- [Axiom Studio]: WARNING: Failed to load bundled JetBrains Mono font. ---"
            )
        else:
            print(
                "--- [Axiom Studio]: Successfully loaded bundled JetBrains Mono font. ---"
            )
    window = AxiomStudio(agent=agent)
    window.show()

    exit_code = app.exec()
    server_thread.join(timeout=2)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
