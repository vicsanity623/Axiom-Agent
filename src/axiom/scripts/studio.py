from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import uvicorn
from PyQt6.Qsci import QsciLexerPython, QsciScintilla
from PyQt6.QtCore import QProcess, Qt, QUrl
from PyQt6.QtGui import QFileSystemModel, QFont, QFontDatabase
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QApplication,
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


class AxiomStudio(QMainWindow):
    """
    The main window for the Axiom Studio IDE, now with a full IDE layout.
    """

    def __init__(self, agent: CognitiveAgent):
        super().__init__()
        self.agent = agent
        self.setWindowTitle(f"Axiom Studio v{__version__}")
        self.setGeometry(100, 100, 1600, 1000)

        self._create_toolbar()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        self.file_browser = QTreeView()
        self.file_system_model = QFileSystemModel()
        self.file_system_model.setRootPath(str(PROJECT_ROOT))
        self.file_browser.setModel(self.file_system_model)
        self.file_browser.setRootIndex(self.file_system_model.index(str(PROJECT_ROOT)))
        self.file_browser.setColumnWidth(0, 250)
        self.file_browser.doubleClicked.connect(self.open_file_in_editor)
        splitter.addWidget(self.file_browser)

        self.workspace = QTabWidget()
        splitter.addWidget(self.workspace)

        self.browser = QWebEngineView()
        url = QUrl("http://127.0.0.1:8000/studio")
        self.browser.setUrl(url)
        self.workspace.addTab(self.browser, "Agent Console")

        self.editor = QsciScintilla()
        self._configure_editor()
        self.workspace.addTab(self.editor, "Code Editor")

        self.console_view = QWebEngineView()
        self.console_view.setUrl(QUrl("http://127.0.0.1:8000/console"))
        self.workspace.addTab(self.console_view, "Console")

        splitter.setSizes([300, 1300])

    def _create_toolbar(self):
        """Creates the native toolbar with lifecycle controls."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        self.play_button = QPushButton("Play Start")
        # --- FIX: Provide a valid, simple stylesheet ---
        self.play_button.setStyleSheet(
            "background-color: #4f46e5; color: white; padding: 5px 15px; border-radius: 5px;"
        )
        self.play_button.clicked.connect(self.start_training_clicked)
        toolbar.addWidget(self.play_button)
        self.stop_button = QPushButton("Stop")
        # --- FIX: Provide a valid, simple stylesheet ---
        self.stop_button.setStyleSheet(
            "background-color: #dc2626; color: white; padding: 5px 15px; border-radius: 5px;"
        )
        self.stop_button.clicked.connect(self.stop_training_clicked)
        self.stop_button.setEnabled(False)
        toolbar.addWidget(self.stop_button)

    def _configure_editor(self):
        """Sets up the QScintilla code editor with Python syntax highlighting."""
        editor_font = QFont()
        editor_font.setFamily("JetBrains Mono, Menlo, Monaco, Courier New, monospace")
        editor_font.setPointSize(12)
        editor_font.setStyleHint(QFont.StyleHint.Monospace)

        self.editor.setFont(editor_font)
        lexer = QsciLexerPython()
        lexer.setDefaultFont(editor_font)
        self.editor.setLexer(lexer)
        self.editor.setMarginsFont(editor_font)
        font_metrics = self.fontMetrics()
        self.editor.setMarginWidth(0, font_metrics.horizontalAdvance("0000"))
        self.editor.setMarginLineNumbers(0, True)
        self.editor.setMarginsBackgroundColor(Qt.GlobalColor.darkGray)
        self.editor.setMarginsForegroundColor(Qt.GlobalColor.lightGray)
        self.editor.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)
        self.editor.setCaretLineVisible(True)
        self.editor.setCaretLineBackgroundColor(Qt.GlobalColor.darkCyan)
        self.editor.setIndentationsUseTabs(False)
        self.editor.setTabWidth(4)
        self.editor.setIndentationGuides(True)
        self.editor.setAutoIndent(True)

    def open_file_in_editor(self, index):
        """Slot to handle double-clicking a file in the QTreeView."""
        path = self.file_system_model.filePath(index)
        file_path = Path(path)
        if file_path.is_file() and file_path.suffix == ".py":
            try:
                content = file_path.read_text(encoding="utf-8")
                self.editor.setText(content)
                self.workspace.setCurrentWidget(self.editor)
            except Exception as e:
                print(f"--- [Axiom Studio]: Error opening file {path}: {e} ---")

    def handle_process_state_change(self, new_state):
        """Re-enables the Play button when a process finishes."""
        if new_state == QProcess.ProcessState.NotRunning:
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def start_training_clicked(self):
        """
        Handles the 'Play Start' button click by injecting the 'axiom-train'
        command into the terminal web view.
        """
        self.workspace.setCurrentWidget(self.console_view)
        # The '\n' is the equivalent of pressing Enter
        js_command = "window.axiom_terminal.write('axiom-train\\n');"
        self.console_view.page().runJavaScript(js_command)

        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_training_clicked(self):
        """
        Handles the 'Stop' button click by injecting the Ctrl+C character
        into the terminal web view.
        """
        if self.workspace.currentWidget() != self.console_view:
            self.workspace.setCurrentWidget(self.console_view)

        # The character \x03 is the ASCII code for Ctrl+C
        js_command = "window.axiom_terminal.write('\\x03');"
        self.console_view.page().runJavaScript(js_command)

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
