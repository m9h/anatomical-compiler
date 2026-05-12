import subprocess
import sys
from pathlib import Path

def test_sbi_dry_run():
    # Run the script in dry-run mode with minimal epochs for speed
    script_path = Path("scripts/12_pollen_inverse_sbi.py")
    cmd = [sys.executable, str(script_path), "--dry-run", "--epochs", "2"]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check for successful execution
    assert result.returncode == 0
    assert "Inverse modeling complete" in result.stdout
    assert Path("figures/pollen_sbi_jacobian.png").exists()
