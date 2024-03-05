from pylablib.devices import Thorlabs


class ThorlabsZStage:
    def __init__(self, device_sn):
        self.stage = Thorlabs.KinesisMotor(device_sn)
        self.stage.open()

    def move_to(self, z):
        self.stage.move_to(z)

    def close(self):
        self.stage.close()

    def get_position(self):
        return self.stage.position

    def get_velocity(self):
        return self.stage.velocity

    def get_status(self):
        return self.stage.status


if __name__ == '__main__':
    print(Thorlabs.list_kinesis_devices())
    device_sn = Thorlabs.list_kinesis_devices()[0][0]
    z_stage = ThorlabsZStage(device_sn)
    print(z_stage.get_position())
    print(z_stage.get_velocity())
    print(z_stage.get_status())
    z_stage.move_to(0)
    print(z_stage.get_position())
    z_stage.close()

