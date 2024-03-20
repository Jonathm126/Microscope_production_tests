from sequence_grabber import SequenceRunner
folder = "C:\\Users\lab1\projects\Microscope_production_tests\CTF\log\\2024-03-20_09-24-59_iris-mm-12"
filename = "C:\\Users\lab1\projects\Microscope_production_tests\CTF\log\\2024-03-20_09-24-59_iris-mm-12\sequence_data.csv"
sequence_runner = SequenceRunner()
sequence_runner.load(filename)
sequence_runner.plot(pdf_output_folder=folder)
