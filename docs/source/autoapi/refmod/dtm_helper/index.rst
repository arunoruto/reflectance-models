refmod.dtm_helper
=================

.. py:module:: refmod.dtm_helper




Module Contents
---------------

.. py:function:: matlab_gradient(img, h)

.. py:function:: dtm2grad(dtm, resolution = 1, normalize = False)

   Computes the gradient of a Digital Terrain Model (DTM).

   :param dtm: The input DTM as a 2D numpy array.
   :type dtm: numpy.ndarray
   :param resolution: The resolution of the DTM. Defaults to 1.
   :type resolution: float, optional
   :param normalize: Flag indicating whether to normalize the gradient vectors. Defaults to False.
   :type normalize: bool, optional

   :returns: The gradient vectors of the DTM.
   :rtype: (numpy.ndarray)

   .. rubric:: Examples

   >>> dtm = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
   >>> dtm2grad(dtm, resolution=0.5, normalize=True)
   array([[[-2.82842712, -2.82842712,  1.41421356],
           [-2.82842712, -2.82842712,  1.41421356],
           [-2.82842712, -2.82842712,  1.41421356]],
          [[-2.82842712, -2.82842712,  1.41421356],
           [-2.82842712, -2.82842712,  1.41421356],
           [-2.82842712, -2.82842712,  1.41421356]],
          [[-2.82842712, -2.82842712,  1.41421356],
           [-2.82842712, -2.82842712,  1.41421356],
           [-2.82842712, -2.82842712,  1.41421356]]])


