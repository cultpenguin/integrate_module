I would like a web-based interface to the **integrate** module, implemented using *Streamlit*.  
All Streamlit files should be located in `@streamlit`.

The interface should consist of four panes:

## 1. `@streamlit@ig_data.py` – Allow analyzing HDF5 files (with extension 'h5') in the current folder. files should be sleected in a listbox, and should show relevant info about the HDF5 file. 
It should also try to identify whether the HDF5file is DATA, PRIOR file (with prior model and/or data), or a POSTERIOR file (with the 'i_use' data set).

## 2. `@streamlit@ig_prior.py` – Allows running `prior_model_layered()`, `prior_model_workbench()`, and `prior_model_workbench_direct()` as defined in `@integrate/integrate.py`.

## 3. `@streamlit@ig_forward.py` – Allows running `forward_gaaem()` as defined in `@integrate/integrate.py`.

The list box of HDF5 data files, should have a toggle that fileter out all files starting with 'PRIOR*' or 'POST*', When a data set is loaded, it should optinally be plotted using ig.plot_geometry()

## 4. `@streamlit@ig_rejection.py` – Allows running `integrate_rejection()` as defined in `@integrate/integrate_rejection.py`.

## 5. `@streamlit@ig_plot.py` – Provides interfaces to run selected functions from `@integrate/integrate_plot.py`.

Where applicable, required `.h5` files should be selected using either a list box or a file upload UI.  
It would be useful if basic information from `f_data_h5`, `d_prior_h5`, and `f_post_h5` could be displayed on-screen to verify that the files are valid before use.

Once an inversion is completed, the application should plot a profile using `ig.plot_profile()`.

Each pane should be runnable individually via:

    streamlit run <filename>.py

Additionally, all panes should be integrated into a main Streamlit application called `integrate_www.py`, where they appear as separate panes within the main interface.

Three types of HDF5 files are considered: f_data_h5, f_prior_h5, and f_post_h5. The format of the se hdf5 files are described in @doc/format.rst

