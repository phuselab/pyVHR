pyVHR.BVP package
=================

Submodules
----------

pyVHR.BVP.BVP module
--------------------

.. automodule:: pyVHR.BVP.BVP
   :members:
   :undoc-members:
   :show-inheritance:

pyVHR.BVP.methods module
------------------------

This module contains a collection of known rPPG methods.

**rPPG METHOD SIGNATURE**:
    - 'signal': RGB signal as float32 ndarray with shape [num_estimators, rgb_channels, num_frames], or a custom signal.
    - '\*\*kargs' [OPTIONAL]: usefull parameters passed to the filter method.

An rPPG method must return a BVP signal as float32 ndarray with shape [num_estimators, num_frames].

.. automodule:: pyVHR.BVP.methods
   :members:
   :undoc-members:
   :show-inheritance:

pyVHR.BVP.filters module
------------------------

This module contains a collection of filter methods.

**FILTER METHOD SIGNATURE**:
    - 'signal': RGB signal as float32 ndarray with shape [num_estimators, rgb_channels, num_frames], or BVP signal as float32 ndarray with shape [num_estimators, num_frames].
    - '\*\*kargs' [OPTIONAL]: usefull parameters passed to the filter method.

A filter method must return a filtered signal with the same shape as the input signal.

.. automodule:: pyVHR.BVP.filters
   :members:
   :undoc-members:
   :show-inheritance:

pyVHR.BVP.utils module
----------------------

.. automodule:: pyVHR.BVP.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: pyVHR.BVP
   :members:
   :undoc-members:
   :show-inheritance:
