from onnxruntime.quantization import CalibrationDataReader

class ONNX_Loader(CalibrationDataReader):
    def __init__(self, calib_loader):
        self.enum_data = None

        self.nhwc_data_list = []
        for i, (images, _) in enumerate(calib_loader):
            images = images.numpy()
            self.nhwc_data_list.append(images)
            break

        self.input_name = "input"
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None