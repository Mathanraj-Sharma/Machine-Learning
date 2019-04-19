#!/usr/bin/env python


import apache_beam as beam
import sys

def my_grep(line, term):
   if line.startswith(term):
      yield line

if __name__ == '__main__':
   p = beam.Pipeline(argv=sys.argv)
   input = '../javahelp/src/main/java/com/google/cloud/training/dataanalyst/javahelp/*.java' #replace withyour input file path. It can be batch data or stream data
   output_prefix = '/tmp/output'
   searchTerm = 'import'

   # find all lines that contain the searchTerm
   (p
      | 'GetJava' >> beam.io.ReadFromText(input) #beam.io.ReadFromPubSub for stream data
      | 'Grep' >> beam.FlatMap(lambda line: my_grep(line, searchTerm) )
      | 'write' >> beam.io.WriteToText(output_prefix)
   )

   p.run().wait_until_finish()


# To run in GCP

#!/usr/bin/env python


import apache_beam as beam

def my_grep(line, term):
   if line.startswith(term):
      yield line

PROJECT='qwiklabs-gcp-865fcc41f9e9c20d' #replace with your project_id
BUCKET='mytestbuck2' #your data bucket name

def run():
   argv = [
      '--project={0}'.format(PROJECT),
      '--job_name=examplejob2',
      '--save_main_session',
      '--staging_location=gs://{0}/staging/'.format(BUCKET),
      '--temp_location=gs://{0}/staging/'.format(BUCKET),
      '--runner=DataflowRunner'
   ]

   p = beam.Pipeline(argv=argv)
   input = 'gs://{0}/javahelp/*.java'.format(BUCKET)
   output_prefix = 'gs://{0}/javahelp/output'.format(BUCKET)
   searchTerm = 'import'

   # find all lines that contain the searchTerm
   (p
      | 'GetJava' >> beam.io.ReadFromText(input)
      | 'Grep' >> beam.FlatMap(lambda line: my_grep(line, searchTerm) )
      | 'write' >> beam.io.WriteToText(output_prefix)
   )

   p.run()

if __name__ == '__main__':
   run()

