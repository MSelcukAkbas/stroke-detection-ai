import tensorflow as tf
import os
max_memory = 12217 # burda 4 ekran 8 normal verdim 16 gb ramde stabil bir şekilde çalışıyor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("GPU cihazları:", physical_devices)
print(tf.sysconfig.get_build_info())
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
if physical_devices:    
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_memory)])
else:
    print("GPU bulunamadı. yapılandırmayı kontrol et ve tekrar dene.")