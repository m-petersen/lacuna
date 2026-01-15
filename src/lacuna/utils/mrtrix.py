"""
MRtrix3 command wrappers for tractography-based lesion network mapping.

Provides Python wrappers around MRtrix3 commands (tckedit, tckmap, mrcalc)
for filtering tractograms by lesion masks and computing disconnection maps.
"""

import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib

from lacuna.utils.cache import make_temp_file


class MRtrixError(Exception):
    """Raised when MRtrix3 commands fail or are not available."""

    pass


def check_mrtrix_available() -> bool:
    """
    Check if MRtrix3 commands are available in the system PATH.

    Returns
    -------
    bool
        True if all required MRtrix3 commands are available

    Raises
    ------
    MRtrixError
        If MRtrix3 is not installed or commands are not in PATH

    Examples
    --------
    >>> from lacuna.utils.mrtrix import check_mrtrix_available
    >>> try:
    ...     check_mrtrix_available()
    ...     print("MRtrix3 is available")
    ... except MRtrixError as e:
    ...     print(f"MRtrix3 not available: {e}")
    """
    required_commands = ["tckedit", "tckmap", "mrcalc"]

    missing_commands = []
    for cmd in required_commands:
        if shutil.which(cmd) is None:
            missing_commands.append(cmd)

    if missing_commands:
        raise MRtrixError(
            f"MRtrix3 commands not found in PATH: {', '.join(missing_commands)}\n"
            f"Please install MRtrix3: https://www.mrtrix.org/download/"
        )

    return True


def run_mrtrix_command(
    command: list[str], check: bool = True, capture_output: bool = False, verbose: bool = False
) -> subprocess.CompletedProcess:
    """
    Execute an MRtrix3 command with proper error handling.

    Parameters
    ----------
    command : list of str
        Command and arguments to execute
    check : bool, default=True
        If True, raises subprocess.CalledProcessError on non-zero exit
    capture_output : bool, default=False
        If True, captures stdout and stderr
    verbose : bool, default=True
        If True, prints the command being executed and allows output to display.
        If False, suppresses both command printing and MRtrix3 output.

    Returns
    -------
    subprocess.CompletedProcess
        Result of command execution

    Raises
    ------
    MRtrixError
        If command execution fails
    """
    if verbose:
        print(f"   Executing: {' '.join(command)}", flush=True)

    # When verbose=False, capture output to suppress it
    should_capture = capture_output or not verbose

    try:
        result = subprocess.run(
            command, check=check, capture_output=should_capture, text=True, encoding="utf-8"
        )
        return result
    except FileNotFoundError as e:
        raise MRtrixError(
            f"MRtrix3 command '{command[0]}' not found. "
            f"Please install MRtrix3: https://www.mrtrix.org/download/"
        ) from e
    except subprocess.CalledProcessError as e:
        raise MRtrixError(f"MRtrix3 command failed: {' '.join(command)}\n{e.stderr}") from e


def filter_tractogram_by_lesion(
    tractogram_path: str | Path,
    lesion_mask: str | Path | nib.Nifti1Image,
    output_path: str | Path | None = None,
    n_jobs: int = 1,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Filter a whole-brain tractogram to streamlines passing through a lesion mask.

    Uses MRtrix3's tckedit with -include to extract only streamlines that
    intersect with the lesion mask.

    Parameters
    ----------
    tractogram_path : str or Path
        Path to whole-brain tractogram (.tck file)
    lesion_mask : str, Path, or nibabel.Nifti1Image
        Lesion mask image or path to NIfTI file
    output_path : str, Path, or None
        Output path for filtered tractogram. If None, creates temp file.
    n_jobs : int, default=1
        Number of threads for MRtrix3 to use
    force : bool, default=False
        Overwrite existing output file
    verbose : bool, default=True
        If True, prints MRtrix3 commands being executed

    Returns
    -------
    Path
        Path to filtered tractogram file

    Raises
    ------
    MRtrixError
        If MRtrix3 command fails
    FileNotFoundError
        If input files don't exist

    Examples
    --------
    >>> from lacuna.utils.mrtrix import filter_tractogram_by_lesion
    >>> filtered_tck = filter_tractogram_by_lesion(
    ...     tractogram_path="whole_brain.tck",
    ...     lesion_mask="lesion.nii.gz",
    ...     output_path="lesion_streamlines.tck",
    ...     n_jobs=8
    ... )

    Notes
    -----
    - Input tractogram must be in same coordinate space as lesion mask (typically MNI152)
    - Output is a .tck file containing only streamlines passing through lesion
    - For large tractograms, this operation can be slow (minutes to hours)
    """
    tractogram_path = Path(tractogram_path)
    if not tractogram_path.exists():
        raise FileNotFoundError(f"Tractogram file not found: {tractogram_path}")

    # Handle lesion mask - save to temp file if needed
    lesion_mask_path = None
    temp_lesion_file = None

    if isinstance(lesion_mask, (str, Path)):
        lesion_mask_path = Path(lesion_mask)
        if not lesion_mask_path.exists():
            raise FileNotFoundError(f"Lesion mask not found: {lesion_mask_path}")
    elif isinstance(lesion_mask, nib.Nifti1Image):
        # Save to temporary file
        temp_lesion_file = make_temp_file(suffix=".nii.gz", delete=False)
        lesion_mask_path = Path(temp_lesion_file.name)
        nib.save(lesion_mask, lesion_mask_path)
    else:
        raise TypeError(
            f"lesion_mask must be str, Path, or nibabel.Nifti1Image, got {type(lesion_mask)}"
        )

    # Determine output path
    if output_path is None:
        temp_output = make_temp_file(suffix=".tck", delete=False)
        output_path = Path(temp_output.name)
    else:
        output_path = Path(output_path)

    # Check if output exists and force flag
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {output_path}\nUse force=True to overwrite"
        )

    try:
        # Build tckedit command
        cmd = [
            "tckedit",
            str(tractogram_path),
            str(output_path),
            "-include",
            str(lesion_mask_path),
            "-nthreads",
            str(n_jobs),
        ]

        if force:
            cmd.append("-force")

        # Execute
        run_mrtrix_command(cmd, verbose=verbose)

        return output_path

    finally:
        # Clean up temporary lesion file if created
        if temp_lesion_file is not None:
            try:
                os.unlink(lesion_mask_path)
            except Exception:
                pass


def compute_tdi_map(
    tractogram_path: str | Path,
    template: str | Path | nib.Nifti1Image,
    output_path: str | Path | None = None,
    n_jobs: int = 1,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Compute Track Density Image (TDI) from a tractogram.

    Uses MRtrix3's tckmap to create a voxel-wise count of streamline endpoints
    or trajectories passing through each voxel.

    Parameters
    ----------
    tractogram_path : str or Path
        Path to tractogram (.tck file)
    template : str, Path, or nibabel.Nifti1Image
        Template image defining output grid, or path to NIfTI file
    output_path : str, Path, or None
        Output path for TDI map. If None, creates temp file.
    n_jobs : int, default=1
        Number of threads for MRtrix3 to use
    force : bool, default=False
        Overwrite existing output file
    verbose : bool, default=True
        If True, prints MRtrix3 commands being executed

    Returns
    -------
    Path
        Path to TDI map file

    Raises
    ------
    MRtrixError
        If MRtrix3 command fails
    FileNotFoundError
        If input files don't exist

    Examples
    --------
    >>> from lacuna.utils.mrtrix import compute_tdi_map
    >>> tdi = compute_tdi_map(
    ...     tractogram_path="lesion_streamlines.tck",
    ...     template="MNI152_T1_2mm.nii.gz",
    ...     output_path="lesion_tdi.nii.gz",
    ...     n_jobs=8
    ... )

    Notes
    -----
    - Template defines the output grid resolution and FOV
    - Uses -contrast tdi for standard track density imaging
    """
    tractogram_path = Path(tractogram_path)
    if not tractogram_path.exists():
        raise FileNotFoundError(f"Tractogram file not found: {tractogram_path}")

    # Handle template - save to temp file if needed
    template_path = None
    temp_template_file = None

    if isinstance(template, (str, Path)):
        template_path = Path(template)
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
    elif isinstance(template, nib.Nifti1Image):
        # Save to temporary file
        temp_template_file = make_temp_file(suffix=".nii.gz", delete=False)
        template_path = Path(temp_template_file.name)
        nib.save(template, template_path)
    else:
        raise TypeError(f"template must be str, Path, or nibabel.Nifti1Image, got {type(template)}")

    # Determine output path
    if output_path is None:
        temp_output = make_temp_file(suffix=".nii.gz", delete=False)
        output_path = Path(temp_output.name)
    else:
        output_path = Path(output_path)

    # Check if output exists and force flag
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {output_path}\nUse force=True to overwrite"
        )

    try:
        # Build tckmap command
        cmd = [
            "tckmap",
            "-contrast",
            "tdi",
            "-template",
            str(template_path),
            str(tractogram_path),
            str(output_path),
            "-nthreads",
            str(n_jobs),
        ]

        if force:
            cmd.append("-force")

        # Execute
        run_mrtrix_command(cmd, verbose=verbose)

        return output_path

    finally:
        # Clean up temporary template file if created
        if temp_template_file is not None:
            try:
                os.unlink(template_path)
            except Exception:
                pass


def compute_disconnection_map(
    lesion_tdi: str | Path | nib.Nifti1Image,
    whole_brain_tdi: str | Path | nib.Nifti1Image,
    output_path: str | Path | None = None,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Compute disconnection map as ratio of lesion TDI to whole-brain TDI.

    Uses MRtrix3's mrcalc to divide lesion TDI by whole-brain TDI, producing
    a voxel-wise disconnection probability map.

    Parameters
    ----------
    lesion_tdi : str, Path, or nibabel.Nifti1Image
        TDI from lesion-filtered tractogram
    whole_brain_tdi : str, Path, or nibabel.Nifti1Image
        TDI from whole-brain tractogram (reference)
    output_path : str, Path, or None
        Output path for disconnection map. If None, creates temp file.
    force : bool, default=False
        Overwrite existing output file
    verbose : bool, default=True
        If True, prints MRtrix3 commands being executed

    Returns
    -------
    Path
        Path to disconnection map file

    Raises
    ------
    MRtrixError
        If MRtrix3 command fails
    FileNotFoundError
        If input files don't exist

    Examples
    --------
    >>> from lacuna.utils.mrtrix import compute_disconnection_map
    >>> disconn_map = compute_disconnection_map(
    ...     lesion_tdi="lesion_tdi.nii.gz",
    ...     whole_brain_tdi="whole_brain_tdi.nii.gz",
    ...     output_path="disconnection_map.nii.gz"
    ... )

    Notes
    -----
    - Values range from 0 (no disconnection) to 1 (complete disconnection)
    - Voxels with zero whole-brain TDI are handled gracefully (set to 0)
    """
    # Handle lesion TDI input
    lesion_tdi_path = None
    temp_lesion_file = None

    if isinstance(lesion_tdi, (str, Path)):
        lesion_tdi_path = Path(lesion_tdi)
        if not lesion_tdi_path.exists():
            raise FileNotFoundError(f"Lesion TDI not found: {lesion_tdi_path}")
    elif isinstance(lesion_tdi, nib.Nifti1Image):
        temp_lesion_file = make_temp_file(suffix=".nii.gz", delete=False)
        lesion_tdi_path = Path(temp_lesion_file.name)
        nib.save(lesion_tdi, lesion_tdi_path)
    else:
        raise TypeError(
            f"lesion_tdi must be str, Path, or nibabel.Nifti1Image, got {type(lesion_tdi)}"
        )

    # Handle whole-brain TDI input
    wb_tdi_path = None
    temp_wb_file = None

    if isinstance(whole_brain_tdi, (str, Path)):
        wb_tdi_path = Path(whole_brain_tdi)
        if not wb_tdi_path.exists():
            raise FileNotFoundError(f"Whole-brain TDI not found: {wb_tdi_path}")
    elif isinstance(whole_brain_tdi, nib.Nifti1Image):
        temp_wb_file = make_temp_file(suffix=".nii.gz", delete=False)
        wb_tdi_path = Path(temp_wb_file.name)
        nib.save(whole_brain_tdi, wb_tdi_path)
    else:
        raise TypeError(
            f"whole_brain_tdi must be str, Path, or nibabel.Nifti1Image, got {type(whole_brain_tdi)}"
        )

    # Determine output path
    if output_path is None:
        temp_output = make_temp_file(suffix=".nii.gz", delete=False)
        output_path = Path(temp_output.name)
    else:
        output_path = Path(output_path)

    # Check if output exists and force flag
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {output_path}\nUse force=True to overwrite"
        )

    try:
        # Build mrcalc command
        cmd = [
            "mrcalc",
            str(lesion_tdi_path),
            str(wb_tdi_path),
            "-divide",
            str(output_path),
        ]

        if force:
            cmd.append("-force")

        # Execute
        run_mrtrix_command(cmd, verbose=verbose)

        return output_path

    finally:
        # Clean up temporary files if created
        for temp_file, temp_path in [
            (temp_lesion_file, lesion_tdi_path),
            (temp_wb_file, wb_tdi_path),
        ]:
            if temp_file is not None:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass


def compute_whole_brain_tdi(
    tractogram_path: str | Path,
    output_1mm: str | Path | None = None,
    output_2mm: str | Path | None = None,
    n_jobs: int = 1,
    force: bool = False,
) -> dict[str, Path | None]:
    """
    Compute whole-brain Track Density Images (TDI) at 1mm and/or 2mm MNI resolution.

    Convenience function that computes TDI maps from a whole-brain tractogram
    using bundled MNI152 templates at different resolutions. This generates the
    reference connectivity maps needed for disconnection analysis.

    Parameters
    ----------
    tractogram_path : str or Path
        Path to whole-brain tractogram (.tck file)
    output_1mm : str, Path, or None, optional
        Output path for 1mm TDI map. If None, 1mm TDI is not computed.
    output_2mm : str, Path, or None, optional
        Output path for 2mm TDI map. If None, 2mm TDI is not computed.
    n_jobs : int, default=1
        Number of threads for MRtrix3 to use
    force : bool, default=False
        Overwrite existing output files

    Returns
    -------
    dict
        Dictionary with keys "tdi_1mm" and "tdi_2mm" containing Path objects
        to the computed TDI maps, or None if not computed.

    Raises
    ------
    MRtrixError
        If MRtrix3 command fails
    FileNotFoundError
        If tractogram file doesn't exist
    ValueError
        If neither output_1mm nor output_2mm is specified

    Examples
    --------
    >>> from lacuna.utils.mrtrix import compute_whole_brain_tdi
    >>>
    >>> # Compute both resolutions
    >>> tdis = compute_whole_brain_tdi(
    ...     tractogram_path="whole_brain.tck",
    ...     output_1mm="tdi_1mm.nii.gz",
    ...     output_2mm="tdi_2mm.nii.gz",
    ...     n_jobs=8
    ... )
    >>> print(f"1mm TDI: {tdis['tdi_1mm']}")
    >>> print(f"2mm TDI: {tdis['tdi_2mm']}")
    >>>
    >>> # Compute only 2mm resolution
    >>> tdis = compute_whole_brain_tdi(
    ...     tractogram_path="whole_brain.tck",
    ...     output_2mm="tdi_2mm.nii.gz",
    ...     n_jobs=8
    ... )

    Notes
    -----
    - Uses bundled FSL MNI152 T1 templates (1mm: 182×218×182, 2mm: 91×109×91)
    - Processing time scales with tractogram size (typically 5-30 minutes)
    - Output files can be large (hundreds of MB for dense tractograms)
    - These TDI maps serve as reference connectivity for disconnection analysis

    See Also
    --------
    compute_tdi_map : Lower-level function for custom TDI computation
    StructuralNetworkMapping : Analysis that uses whole-brain TDI as reference
    """
    from lacuna.assets.templates import load_template

    if output_1mm is None and output_2mm is None:
        raise ValueError("At least one of output_1mm or output_2mm must be specified")

    tractogram_path = Path(tractogram_path)
    if not tractogram_path.exists():
        raise FileNotFoundError(f"Tractogram file not found: {tractogram_path}")

    results = {"tdi_1mm": None, "tdi_2mm": None}

    # Compute 1mm TDI if requested
    if output_1mm is not None:
        print("Computing 1mm whole-brain TDI...")
        template_1mm = load_template("MNI152NLin2009cAsym_res-1")
        output_1mm_path = Path(output_1mm)

        if output_1mm_path.exists() and not force:
            print(f"✓ 1mm TDI already exists: {output_1mm_path}")
            results["tdi_1mm"] = output_1mm_path
        else:
            tdi_1mm_path = compute_tdi_map(
                tractogram_path=tractogram_path,
                template=template_1mm,
                output_path=output_1mm_path,
                n_jobs=n_jobs,
                force=force,
            )
            results["tdi_1mm"] = tdi_1mm_path
            print(f"✓ 1mm TDI saved to: {tdi_1mm_path}")

    # Compute 2mm TDI if requested
    if output_2mm is not None:
        print("Computing 2mm whole-brain TDI...")
        template_2mm = load_template("MNI152NLin2009cAsym_res-2")
        output_2mm_path = Path(output_2mm)

        if output_2mm_path.exists() and not force:
            print(f"✓ 2mm TDI already exists: {output_2mm_path}")
            results["tdi_2mm"] = output_2mm_path
        else:
            tdi_2mm_path = compute_tdi_map(
                tractogram_path=tractogram_path,
                template=template_2mm,
                output_path=output_2mm_path,
                n_jobs=n_jobs,
                force=force,
            )
            results["tdi_2mm"] = tdi_2mm_path
            print(f"✓ 2mm TDI saved to: {tdi_2mm_path}")

    return results
