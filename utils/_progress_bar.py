from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

class ProgressBar:
    """
    Wrapper around rich.progress.Progress to show progress bars for training and trials.
    """  
    pbar = None  
    trial = False
    trials_id = None
    epochs_id = None
    
        
    def __init__(self, mode = "train"):
        self.mode = mode
        if self.mode == "trial":
            ProgressBar.trial = True
        
    def __enter__(self):
        if not (ProgressBar.trial and self.mode == "train"):
            ProgressBar.pbar = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    SpinnerColumn(spinner_name="runner"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(),
                    refresh_per_second=5,
                )
            if ProgressBar.trial:
                ProgressBar.trials_id = ProgressBar.pbar.add_task("[bold blue]Trials...", total=None)
            ProgressBar.epochs_id = ProgressBar.pbar.add_task("[bold green]Epochs...", total=None)
            ProgressBar.train_id = ProgressBar.pbar.add_task("[cyan]Train epoch...", total=None)
            ProgressBar.val_id = ProgressBar.pbar.add_task("[cyan]Validation epoch...", total=None)
            ProgressBar.pbar.__enter__()
        return self
    
    def __exit__(self, *args):
        if not (ProgressBar.trial and self.mode == "train"):
            ProgressBar.trial = False
            ProgressBar.pbar.refresh()  
            ProgressBar.pbar.__exit__(*args)
            
            
def create_progress_bar():
    pbar = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    SpinnerColumn(spinner_name="runner"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(),
                    refresh_per_second=5,
                )
    
    epochs_id = pbar.add_task("[bold green]Epochs...", total=None)
    train_id = pbar.add_task("[cyan]Train epoch...", total=None)
    val_id = pbar.add_task("[cyan]Validation epoch...", total=None)

class ProgressBar2:
    """
    Wrapper around rich.progress.Progress
    """  
    pbar = None  
    trial = False
    trials_id = None
    epochs_id = None
    
        
    def __init__(self, mode = "train"):
        self.mode = mode
        if self.mode == "trial":
            ProgressBar.trial = True
        
    def __enter__(self):
        if not (ProgressBar.trial and self.mode == "train"):
            ProgressBar.pbar = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    SpinnerColumn(spinner_name="runner"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(),
                    refresh_per_second=5,
                )
            if ProgressBar.trial:
                ProgressBar.trials_id = ProgressBar.pbar.add_task("[bold blue]Trials...", total=None)
            ProgressBar.epochs_id = ProgressBar.pbar.add_task("[bold green]Epochs...", total=None)
            ProgressBar.train_id = ProgressBar.pbar.add_task("[cyan]Train epoch...", total=None)
            ProgressBar.val_id = ProgressBar.pbar.add_task("[cyan]Validation epoch...", total=None)
            ProgressBar.pbar.__enter__()
        return self
    
    def __exit__(self, *args):
        if not (ProgressBar.trial and self.mode == "train"):
            ProgressBar.trial = False
            ProgressBar.pbar.refresh()  
            ProgressBar.pbar.__exit__(*args)
            

from collections import deque
from rich.console import ConsoleRenderable, Group, RichCast
from rich.progress import Progress
from rich.table import Table

class CustomProgress(Progress):
    def __init__(self, table_max_rows: int, *args, **kwargs) -> None:
        self.results = deque(maxlen=table_max_rows)
        self.update_table()
        super().__init__(TextColumn("[progress.description]{task.description}"),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        SpinnerColumn(spinner_name="runner"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TextColumn("•"),
                        TimeElapsedColumn(),
                        TextColumn("•"),
                        TimeRemainingColumn(),
                        *args, **kwargs)

    def update_table(self, result: tuple[str] | None = None):
        if result is not None:
            self.results.append(result)
        table = Table()
        table.add_column("Epoch")
        table.add_column("Train loss", width=20)
        table.add_column("Val loss", width=20)

        for row_cells in self.results:
            table.add_row(*row_cells)

        self.table = table

    def get_renderable(self) -> ConsoleRenderable | RichCast | str:
        renderable = Group(self.table, *self.get_renderables())
        return renderable