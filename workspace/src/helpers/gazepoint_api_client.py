# Client API for the OpenGaze API
# based on:
# PyOpenGaze: Python wrapper for the OpenGaze API.
# author: Edwin Dalmaijer
# email: edwin.dalmaijer@psy.ox.ac.uk
# Version 1 (27-Apr-2016)
# https://github.com/esdalmaijer/PyOpenGaze 

# Last update by: Milena Borczak 
# Date: 12.06.2025
# Description: Update client to 2.8 revision, save results to csv file instead of tsv. 


import os
import copy
import time
import socket
import datetime
import lxml.etree
import argparse
from multiprocessing import Queue
from threading import Event, Lock, Thread



# The OpenGazeTracker class communicates to the GazePoint Server through
# a TCP/IP socket.
class OpenGazeTracker:

	def __init__(self, ip='127.0.0.1', port=4242, logfile='default.csv', debug=False):
		
		"""The OpenGazeConnection class communicates to the GazePoint
		server through a TCP/IP socket. Incoming samples will be written
		to a log at the specified path.
		
		Keyword Arguments
		
		ip	-	The IP address of the computer that is running the
				OpenGaze server. This will usually be the localhost at
				127.0.0.1. Type: str. Default = '127.0.0.1'
		
		port	-	The port number that the OpenGaze server is on; usually
				this will be 4242. Type: int. Default = 4242
		
		logfile	-	The path to the intended log file, including a
					file extension ('.csv'). Type: str. Default = 
					'default.csv'

		debug	-	Boolean that determines whether DEBUG mode should be
				active (True) or not (False). In DEBUG mode, all sent
				and received messages are logged to a file. Type: bool.
				Default = False
		"""
		
		# DEBUG
		self._debug = debug
		# Open a new debug file.
		if self._debug:
			dt = time.strftime("%Y-%m-%d_%H-%M-%S")
			self._debuglog = open('debug_%s.txt' % (dt), 'w')
			self._debuglog.write("OPENGAZE PYTHON DEBUG LOG %s\n" % (dt))
			self._debugcounter = 0
			self._debugconsolidatefreq = 100
		
		# CONNECTION
		# Save the ip and port numbers.
		self.host = ip
		self.port = port
		# Start a new TCP/IP socket. It is curcial that it has a timeout,
		# as timeout exceptions will be handled gracefully, and are in fact
		# necessary to prevent the incoming Thread from freezing.
		#print(f"Connecting to {self.host} ({self.port})...")
		self._debug_print(f"Connecting to {self.host} ({self.port})...")

		self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

		try:
			self._sock.connect((self.host, self.port))
		except socket.error as e:
			msg = f"Failed to connect to {self.host}:{self.port} {e}"
			self._debug_print(msg)
			raise ConnectionError(msg)
		
		#self._sock.connect((self.host, self.port))

		self._sock.settimeout(1.0)
		#print("Successfully connected!")
		self._debug_print("Successfully connected!")

		self._maxrecvsize = 4096
		# Create a socket Lock to prevent simultaneous access.
		self._socklock = Lock()
		# Create an event that should remain set until the connection is
		# closed. (This is what keeps the Threads running.)
		self._connected = Event()
		self._connected.set()
		# Set the current calibration point.
		self._current_calibration_point = None
		
		# LOGGING
		self._debug_print(f"Opening new logfile '{logfile}'")
		# Open a new log file.
		self._logfile = open(logfile, 'w')
		# Write the header to the log file.
		self._logheader = ['CNT', 'TIME', 'TIME_TICK',
			'FPOGX', 'FPOGY', 'FPOGS', 'FPOGD', 'FPOGID', 'FPOGV',
			'LPOGX', 'LPOGY', 'LPOGV',
			'RPOGX', 'RPOGY', 'RPOGV',
			'BPOGX', 'BPOGY', 'BPOGV',
			'APOGX', 'APOGY', 'APOGV',
			'LPCX', 'LPCY', 'LPD', 'LPS', 'LPV',
			'RPCX', 'RPCY', 'RPD', 'RPS', 'RPV',
			'LEYEX', 'LEYEY', 'LEYEZ', 'LPUPILD', 'LPUPILV',
			'REYEX', 'REYEY', 'REYEZ', 'RPUPILD', 'RPUPILV',
			'CX', 'CY', 'CS',
			'KB', 'KBS',
			'BKID', 'BKDUR', 'BKPMIN',
			'LPMM', 'LPMMV', 'RPMM', 'RPMMV',
			'USER']
		self._n_logvars = len(self._logheader)
		self._logfile.write(';'.join(self._logheader) + '\n')
		self._logheader_index = {varname: idx for idx, varname in enumerate(self._logheader)}

		# The log is consolidated (written to the disk) every N samples.
		# This requires an internal counter (because we can't be sure the
		# user turned on the 'CNT' sample counter), and a property that
		# determines the consolidation frequency. This frequency can also
		# be set to None, to never consolidate automatically.
		self._logcounter = 0
		self._log_consolidation_freq = 60
		# Start a Queue for samples that need to be logged.
		self._logqueue = Queue()
		# Set an event that is set while samples should be logged, and
		# unset while they shouldn't.
		self._logging = Event()
		self._logging.set()
		# Set an event that signals is set when the logfile is ready to
		# be closed.
		self._log_ready_for_closing = Event()
		self._log_ready_for_closing.clear()
		# Start a Thread that writes queued samples to the log file.
		self._logthread = Thread(target=self._process_logging, name='PyGaze_OpenGazeConnection_logging', args=[])
		
		# INCOMING
		# Start a new dict for the latest incoming messages, and for
		# incoming acknowledgements.
		self._incoming = {}
		self._acknowledgements = {}
		# Create a Lock for the incoming message and acknowledgement dicts.
		self._inlock = Lock()
		self._acklock = Lock()
		# Create an empty string for the current unfinished message. This
		# is to prevent half a message being parsed when it is cut off
		# between two 'self._sock.recv' calls.
		self._unfinished = ''
		# Start a new Thread that processes the incoming messages.
		self._inthread = Thread(target=self._process_incoming, name='PyGaze_OpenGazeConnection_incoming', args=[])
		
		# OUTGOING
		# Start a new outgoing Queue (Thread safe, woop!).
		self._outqueue = Queue()
		# Set an event that is set when all queued outgoing messages have
		# been processed.
		self._sock_ready_for_closing = Event()
		self._sock_ready_for_closing.clear()
		# Create a new Thread that processes the outgoing queue.
		self._outthread = Thread(target=self._process_outgoing, name='PyGaze_OpenGazeConnection_outgoing', args=[])
		# Create a dict that will keep track of at what time which command
		# was sent.
		self._outlatest = {}
		# Create a Lock to prevent simultaneous access to the outlatest
		# dict.
		self._outlock = Lock()
		
		# RUN THREADS
		# Set a signal that will kill all Threads when they receive it.
		self._thread_shutdown_signal = 'STOP_THREAD'
		# Start the threads.
		self._debug_print("Starting the logging thread.")
		self._logthread.start()
		self._debug_print("Starting the incoming thread.")
		self._inthread.start()
		self._debug_print("Starting the outgoing thread.")
		self._outthread.start()
		
		# SET UP LOGGING
		# Wait for a bit to allow the Threads to start.
		time.sleep(0.5)
		# Enable the tracker to send ALL the things.
		self.enable_send_counter(True)
		self.enable_send_cursor(True)
		self.enable_send_eye_left(True)
		self.enable_send_eye_right(True)
		self.enable_send_pog_best(True)
		self.enable_send_pog_fix(True)
		self.enable_send_pog_left(True)
		self.enable_send_pog_right(True)
		self.enable_send_pupil_left(True)
		self.enable_send_pupil_right(True)
		self.enable_send_time(True)
		self.enable_send_time_tick(True)
		self.enable_send_user_data(True)
		self.enable_send_kb(True)
		self.enable_send_blink(True)
		self.enable_send_pupilmm(True)
		# Reset the user-defined variable.
		self.user_data("0")

	def calibrate(self):
		
		"""
		Calibrates the eye tracker
		Runs a full eye tracker calibration procedure.

		Steps:
		1. Clears previous calibration results.
		2. Shows the calibration screen.
		3. Starts calibration.
		4. Waits until calibration results are available.
		5. Hides the calibration screen.
		6. Returns the calibration result as a list of point dictionaries.
		"""

		self.clear_calibration_result()
		self.calibrate_show(True)
		self.calibrate_start(True)

		# Wait until results are available
		result = None
		while result is None:
			result = self.get_calibration_result()
			time.sleep(0.1)

		self.calibrate_show(False)
		return result
	
	
	def sample(self):

		"""
		Returns the current (x, y) gaze sample coordinates if available.
		If the sample is not available, returns (None, None).
		"""

		with self._inlock:
			rec = self._incoming.get('REC', {})
			no_id = rec.get('NO_ID', {})
			try:
				x = float(no_id['BPOGX'])
				y = float(no_id['BPOGY'])
			except (KeyError, ValueError, TypeError):
				x, y = None, None

		return x, y
	
	
	def pupil_size(self):
		"""
		Returns the current average pupil size, based on left and/or right eye data.

		If no valid data is available, returns None.
		"""
		with self._inlock:
			rec = self._incoming.get('REC', {}).get('NO_ID', {})
			l_valid = rec.get('LPV') == '1'
			r_valid = rec.get('RPV') == '1'

			total = 0.0
			count = 0

			try:
				if l_valid:
					total += float(rec['LPS'])
					count += 1
				if r_valid:
					total += float(rec['RPS'])
					count += 1
			except (KeyError, ValueError, TypeError):
				return None

		return total / count if count > 0 else None
	

	def log(self, message):
		"""
		Logs a message to the log file. ONLY CALL THIS WHILE RECORDING DATA!

		Args:
			message (str): Message to log.
		"""
		start_counter = self._logcounter
		self.user_data(message)

		while self._logcounter <= start_counter:
			time.sleep(0.0001)  # 1 ms pause 

		self.user_data("0")

	
	def start_recording(self):
		
		"""Start writing data to the log file.
		"""
		
		self.enable_send_data(True)
	
	def stop_recording(self):
		
		"""Pause writing data to the log file.
		"""
		
		self.enable_send_data(False)


	def _debug_print(self, msg):
		print(msg)
		if not self._debug or not hasattr(self, '_debuglog'):
			return

		try:
			timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
			self._debuglog.write(f"{timestamp}: {msg}\n")
			self._debugcounter += 1

			if self._debugcounter % self._debugconsolidatefreq == 0:
				self._debuglog.flush()
				os.fsync(self._debuglog.fileno())

		except Exception as e:
			# fallback log to console if file logging fails
			print(f"[DEBUG_LOG ERROR] {e}")



	def _format_msg(self, command, ID, values=None):
		"""
		Formats an XML-like message string.

		Example output:
			<GET ID="SOME_ID" PARAM1="VALUE1" PARAM2="VALUE2" />

		Args:
			command (str): Command name (e.g., 'GET', 'SET')
			ID (str): Message ID
			values (list[tuple[str, str]], optional): List of (param, value) pairs

		Returns:
			str: Formatted message string
		"""
		attrs = [f'{par.upper()}="{val}"' for par, val in values] if values else []
		attr_str = ' '.join(attrs)
		return f'<{command.upper()} ID="{ID.upper()}" {attr_str}/>\r\n'



	def _log_consolidation(self):
		"""
		Forces the log buffer to be written:
		- from Python internal buffer to OS-level (flush),
		- and from OS buffer to disk (fsync).
		"""
		try:
			self._logfile.flush()  # Python -> OS buffer
			os.fsync(self._logfile.fileno())  # OS buffer -> disk
		except (ValueError, OSError) as e:
			# Optional: log the error or handle gracefully
			self._debug_print(f"Log consolidation failed: {e}")
	

	def _log_sample(self, sample):
		"""
		Writes a log line based on the given sample dictionary.

		Each key in the sample is matched to the appropriate position
		in the log header. Missing fields remain empty.
		"""
		# Pre-fill line with empty strings
		line = self._n_logvars * ['']

		for varname, value in sample.items():
			idx = self._logheader_index.get(varname)
			if idx is not None:
				line[idx] = str(value)

		self._logfile.write(';'.join(line) + '\n')


	def _parse_msg(self, xml):
		"""
		Parses an XML message string and returns its tag and attributes.

		Args:
			xml (str): XML string to parse.

		Returns:
			Tuple[str, dict]: (tag name, attributes dictionary)
		
		Raises:
			etree.XMLSyntaxError: If the XML is not well-formed.
		"""
		try:
			element = lxml.etree.fromstring(xml)
			return element.tag, element.attrib
		except lxml.etree.XMLSyntaxError as e:
			# Optional: log the error or re-raise
			self._debug_print(f"Failed to parse XML: {e}")
			raise


	
	def _process_logging(self):
		"""
		Thread loop that processes the logging queue:
		- Retrieves samples from queue
		- Writes them to log
		- Periodically flushes log to disk
		- Stops gracefully on shutdown signal
		??? czat 
		"""
		
		self._debug_print("Logging Thread started.")
		
		while not self._log_ready_for_closing.is_set():
			try:

				# Get a new sample from the Queue.
				sample = self._logqueue.get()
				
				# Check if this is the shutdown signal.
				if sample == self._thread_shutdown_signal:
					# Signal that we're done logging all samples.
					self._log_ready_for_closing.set()
					# Break the while loop.
					break
				
				# Log the sample.
				self._log_sample(sample)
				
				# Consolidate the log if necessary.
				if self._logcounter % self._log_consolidation_freq == 0:
					self._log_consolidation()

				# Increment the counter.
				self._logcounter += 1

			except Exception as e:
				# Optional: log or handle logging-related errors
				self._debug_print(f"[log thread] Logging error: {e}")
		
		self._debug_print("Logging Thread ended.")
		return
	

	def _process_incoming(self):
		"""
		Thread function for continuously receiving and processing messages
		from the OpenGaze Server. It updates the internal _incoming dictionary
		and queues 'REC' messages for logging if recording is active.
		??? czat 
		"""

		self._debug_print("Incoming Thread started.")
		
		while self._connected.is_set():

			# # Lock the socket to prevent other Threads from simultaneously
			# # accessing it.

			# Get new messages from the OpenGaze Server.
			timeout = False
			with self._socklock:
				try:
					instring = self._sock.recv(self._maxrecvsize)
				except socket.timeout:
					timeout = True
				# Get a received timestamp.
				t = time.time()
			
			# Skip further processing if no new message came in.
			if timeout:
				self._debug_print("socket recv timeout")
				continue
			
			self._debug_print(f"Raw instring: {instring!r}")
			

			# Split the messages (they are separated by '\r\n').
			messages = instring.decode("utf-8").split('\r\n')
			

			# Check if there is currently an unfinished message.
			if self._unfinished:
				# Combine the currently unfinished message and the
				# most recent incoming message.
				messages[0] = self._unfinished + messages[0]
				# Reset the unfinished message.
				self._unfinished = ''
			# Check if the last message was actually complete.

			if not messages[-1].endswith('/>'):
				self._unfinished = messages.pop(-1)
	
			
			# Run through all messages.
			for msg in messages:
				self._debug_print(f"Incoming: {msg!r}")
				# Parse the message.
				command, msgdict = self._parse_msg(msg)
				# Check if the incoming message is an acknowledgement.
				# Acknowledgements are also stored in a different dict,
				# which is used to monitor whether sent messages are
				# properly received.
				if command == 'ACK':
					with self._acklock:
						self._acknowledgements[msgdict['ID']] = t

				with self._inlock:
					cmd_store = self._incoming.setdefault(command, {})
					msg_id = msgdict.get('ID', 'NO_ID')
					msg_store = cmd_store.setdefault(msg_id, {})
					msg_store['t'] = t

					for par, val in msgdict.items():
						msg_store[par] = val

					# Queue 'REC' messages if logging is on
					if command == 'REC' and self._logging.is_set():
						self._logqueue.put(copy.deepcopy(msg_store))
	
		
		self._debug_print("Incoming Thread ended.")
		return


	def _process_outgoing(self):
		"""
		Thread function for sending messages to the OpenGaze server.
		It listens to the outgoing message queue and sends them over the socket.
		Shuts down gracefully when receiving the shutdown signal.
		czat ???
		"""
		
		self._debug_print("Outgoing Thread started.")
		
		while not self._sock_ready_for_closing.is_set():
			try: 
				# Get a new command from the Queue.
				msg = self._outqueue.get()
				
				# Check if this is the shutdown signal.
				if msg == self._thread_shutdown_signal:
					# Signal that we're done processing all the outgoing
					# messages.
					self._sock_ready_for_closing.set()
					# Break the while loop.
					break
				
				self._debug_print("Outgoing: %r" % (msg))

				# Lock the socket to prevent other Threads from simultaneously
				# accessing it.
				# Send the command to the OpenGaze Server.
				t = time.time()

				with self._socklock:
					try:
						self._sock.send(msg.encode('utf-8'))
					except Exception as e:
						self._debug_print(f"Error sending message: {e}")
						continue
				
				# Save timestamp of message
				with self._outlock:
					# Store a timestamp for the latest outgoing message.
					self._outlatest[msg] = t
			
			except Exception as e:
				self._debug_print(f"Outgoing thread error: {e}")
		
		self._debug_print("Outgoing Thread ended.")
		return
	

	def _send_message(self, command, ID, values=None, wait_for_acknowledgement=True, resend_timeout=3.0, maxwait=9.0):
		"""
		Sends a message to the OpenGaze server and optionally waits for an acknowledgement.

		Args:
			command (str): Message command (e.g. 'GET', 'SET').
			ID (str): Unique ID for the message.
			values (list): Optional list of (key, value) tuples.
			wait_for_acknowledgement (bool): Whether to wait for ACK.
			resend_timeout (float): How long to wait before resending (in seconds).
			maxwait (float): Maximum time to wait for ACK (in seconds).

		Returns:
			(acknowledged, timeout): Tuple of booleans.
		"""
		
		# Format a message in an XML format that the Open Gaze API needs.
		msg = self._format_msg(command, ID, values=values)

		# Run until the message is acknowledged or a timeout occurs (or
		# break if we're not supposed to wait for an acknowledgement.)
		timeout = False
		acknowledged = False
		t0 = time.time()

		while not acknowledged and not timeout:

			# Add the command to the outgoing Queue.
			self._debug_print(f"Outqueue add: {msg!r}")
			self._outqueue.put(msg)

			if not wait_for_acknowledgement:
				break

			sent = False
			t_sent_window_start = time.time()

			# Wait until an acknowledgement comes in.
			while time.time() - t_sent_window_start < resend_timeout and not acknowledged:
				if not sent:
					# Check the outgoing queue for the sent message to
					# appear.
					with self._outlock:
						if msg in self._outlatest:
							t_sent = self._outlatest[msg]
							sent = True
							self._debug_print(f"Outqueue sent: {msg!r}")
					time.sleep(0.001)
				else:
					# Check the incoming queue for the expected
					# acknowledgement. (NOTE: This does not check
					# whether the values of the incoming acknowlement
					# match the sent message. Ideally, they should.)
					with self._acklock:
						ack_time = self._acknowledgements.get(ID)
						if ack_time and ack_time >= t_sent:
							acknowledged = True
							self._debug_print(f"Outqueue acknowledged: {msg!r}")
					time.sleep(0.001)
		
				# Check if there is a timeout.
				if not acknowledged and time.time() - t0 > maxwait:
					timeout = True
					break

			# If we're not supposed to wait for an acknowledgement, break
			# the while loop.
	
		return acknowledged, timeout


	def close(self):
		"""Closes the connection to the tracker, closes the log files, and
		ends the Threads that process the incoming and outgoing messages,
		and the logging of samples.
		"""

		self.user_data('0')
		# Unset the self._connected event to stop the incoming Thread.
		self._debug_print("Unsetting the connection event")
		self._connected.clear()

		# Queue the stop signal to stop the outgoing and logging Threads.
		self._debug_print("Adding stop signal to outgoing Queue")
		self._outqueue.put(self._thread_shutdown_signal)
		self._debug_print("Adding stop signal to logging Queue")
		self._logqueue.put(self._thread_shutdown_signal)

		try:
			
			# Close the socket connection to the OpenGaze server.
			self._debug_print("Waiting for the socket to close...")
			if not self._sock_ready_for_closing.wait(timeout=5):
				self._debug_print("Timeout waiting for socket to close")
			self._sock.close()
			self._debug_print("Socket connection closed!")

			# Wait for the log Queue to be fully processed.
			self._debug_print("Waiting for the log to close...")
			if not self._log_ready_for_closing.wait(timeout=5):
				self._debug_print("Timeout waiting for log to close")
		

			# Close the log file.
			self._logfile.close()
			self._debug_print("Log closed!")

		finally:
			# Join the Threads.
			self._debug_print("Waiting for the Threads to join...")
			self._outthread.join()
			self._debug_print("Outgoing Thread joined!")
			self._inthread.join()
			self._debug_print("Incoming Thread joined!")
			self._logthread.join()
			self._debug_print("Logging Thread joined!")

			# Close the DEBUG log.
			if self._debug:
				self._debuglog.write("END OF DEBUG LOG")
				self._debuglog.close()


	def enable_send_data(self, state):
		
		"""Start (state=True) or stop (state=False) the streaming of data
		from the server to the client.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_DATA', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	

	def enable_send_counter(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		the send counter in the data record string.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_COUNTER', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout


	def enable_send_time(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		the send time in the data record string.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_TIME', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	

	def enable_send_time_tick(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		the send time tick in the data record string.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_TIME_TICK', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	

	def enable_send_pog_fix(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		the point of gaze as determined by the tracker's fixation filter in
		the data record string.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_POG_FIX', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout


	def enable_send_pog_left(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		the point of gaze of the left eye in the data record string.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_POG_LEFT', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout


	def enable_send_pog_right(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		the point of gaze of the right eye in the data record string.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_POG_RIGHT', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout


	def enable_send_pog_best(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		the 'best' point of gaze in the data record string. This is based
		on the average of the left and right POG if both eyes are available,
		or on the value of the one available eye.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_POG_BEST', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout


	def enable_send_pupil_left(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		pupil data on the left eye in the data record string. This data
		consists of the following:
		LPCX: The horizontal coordinate of the left eye pupil in the camera
			image, as a fraction of the camera size.
		LPCY: The vertical coordinate of the left eye pupil in the camera
			image, as a fraction of the camera size.
		LPD:  The left eye pupil's diameter in pixels.
		LPS:  The scale factor of the left eye pupil (unitless). Value
			equals 1 at calibration depth, is less than 1 when the user
			is closer to the eye tracker and greater than 1 when the user
			is further away.
		LPV:  The valid flag with a value of 1 if the data is valid, and 0
			if it is not.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_PUPIL_LEFT', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout


	def enable_send_pupil_right(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		pupil data on the right eye in the data record string. This data
		consists of the following:
		RPCX: The horizontal coordinate of the right eye pupil in the camera
			image, as a fraction of the camera size.
		RPCY: The vertical coordinate of the right eye pupil in the camera
			image, as a fraction of the camera size.
		RPD:  The right eye pupil's diameter in pixels.
		RPS:  The scale factor of the right eye pupil (unitless). Value
			equals 1 at calibration depth, is less than 1 when the user
			is closer to the eye tracker and greater than 1 when the user
			is further away.
		RPV:  The valid flag with a value of 1 if the data is valid, and 0
			if it is not.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_PUPIL_RIGHT', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout


	def enable_send_eye_left(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		3D data on left eye in the data record string. This data consists
		of the following:
		LEYEX:   The horizontal coordinate of the left eye in 3D space with
			   respect to the camera focal point, in meters.
		LEYEY:   The vertical coordinate of the left eye in 3D space with
			   respect to the camera focal point, in meters.
		LEYEZ:   The depth coordinate of the left eye in 3D space with
			   respect to the camera focal point, in meters.
		LPUPILD: The diameter of the left eye pupil in meters.
		LPUPILV: The valid flag with a value of 1 if the data is valid, and
			   0 if it is not.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_EYE_LEFT', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout


	def enable_send_eye_right(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		3D data on right eye in the data record string. This data consists
		of the following:
		REYEX:   The horizontal coordinate of the right eye in 3D space with
			   respect to the camera focal point, in meters.
		REYEY:   The vertical coordinate of the right eye in 3D space with
			   respect to the camera focal point, in meters.
		REYEZ:   The depth coordinate of the right eye in 3D space with
			   respect to the camera focal point, in meters.
		RPUPILD: The diameter of the right eye pupil in meters.
		RPUPILV: The valid flag with a value of 1 if the data is valid, and
			   0 if it is not.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_EYE_RIGHT', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout


	def enable_send_cursor(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		data on the mouse cursor in the data record string. This data
		consists of the following:
		CX:   The horizontal coordinate of the mouse cursor, as a percentage
			of the screen resolution.
		CY:   The vertical coordinate of the mouse cursor, as a percentage
			of the screen resolution.
		CS:   The mouse cursor state, 0 for steady state, 1 for left button
			down, 2 for rigght button down.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_CURSOR', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout


	def enable_send_user_data(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		user-defined variables in the data record string. User-defined
		variables can be set with the 'user_data' method.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_USER_DATA', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	
	def enable_send_kb(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		kb in the data record string. 
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_KB', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	
	def enable_send_blink(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		blink in the data record string. 
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_BLINK', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	
	def enable_send_pupilmm(self, state):
		
		"""Enable (state=True) or disable (state=False) the inclusion of
		pupil mm in the data record string. 
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'ENABLE_SEND_PUPILMM', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout


	def calibrate_start(self, state):
		
		"""Starts (state=1) or stops (state=0) the calibration procedure.
		Make sure to call the 'calibrate_show' function beforehand, or to
		implement your own calibration visualisation; otherwise a call to
		this function will make the calibration run in the background.
		"""
		
		# Reset the current calibration point.
		self._current_calibration_point = 0 if state else None

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'CALIBRATE_START', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	

	def calibrate_show(self, state):
		
		"""Shows (state=1) or hides (state=0) the calibration window on the
		tracker's display window. While showing the calibration window, you
		can call 'calibrate_start' to run the calibration procedure.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'CALIBRATE_SHOW', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout

	
	def calibrate_timeout(self, value):
		
		"""Set the duration of the calibration point (not including the
		animation time) in seconds. The value can be an int or a float.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'CALIBRATE_TIMEOUT', values=[('VALUE', float(value))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
		

	def calibrate_delay(self, value):
		
		"""Set the duration of the calibration animation (before
		calibration at a point begins) in seconds. The value can be an int
		or a float.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'CALIBRATE_DELAY', values=[('VALUE', float(value))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	

	def calibrate_result_summary(self):
		
		"""Returns a summary of the calibration results, which consists of
		the following values:
		AVE_ERROR:    Average error over all calibrated points.
		VALID_POINTS: Number of successfully calibrated points.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('GET', 'CALIBRATE_RESULT_SUMMARY', values=None, wait_for_acknowledgement=True)
		
		
		# Return the results.
		if acknowledged:
				with self._inlock:
					data = self._incoming['ACK']['CALIBRATE_RESULT_SUMMARY']
					ave_error = copy.copy(data.get('AVE_ERROR'))
					valid_points = copy.copy(data.get('VALID_POINTS'))
					return ave_error, valid_points
		return None, None
		
	
	def calibrate_clear(self):
		
		"""Clear the internal list of calibration points.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'CALIBRATE_CLEAR', values=None, wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout


	def calibrate_reset(self):
		
		"""Reset the internal list of calibration points to the default
		values.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'CALIBRATE_RESET', values=None, wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout


	def calibrate_addpoint(self, x, y):
		
		"""Add a calibration point at the passed horizontal (x) and
		vertical (y) coordinates. These coordinates should be as a
		proportion of the screen resolution, where (0,0) is the top-left,
		(0.5,0.5) is the screen centre, and (1,1) is the bottom-right.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'CALIBRATE_ADDPOINT', values=[('X', x), ('Y', y)], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	
	
	def get_calibration_points(self):
		
		"""Returns a list of the current calibration points.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('GET', 'CALIBRATE_ADDPOINT', values=None, wait_for_acknowledgement=True)

		# Return the result.
		if acknowledged:
			with self._inlock:
				data = self._incoming['ACK']['CALIBRATE_ADDPOINT']
				num_points = data.get('PTS', 0)
				points = [
					(float(data[f'X{i+1}']), float(data[f'Y{i+1}']))
					for i in range(num_points)
				]
				return points
		return None


	def clear_calibration_result(self):
		
		"""Clears the internally stored calibration result.
		"""
		
		# Clear the calibration results.
		with self._inlock:
			cal_section = self._incoming.get('CAL')
			if cal_section and 'CALIB_RESULT' in cal_section:
				cal_section.pop('CALIB_RESULT')

	

	def get_calibration_result(self):
		
		"""Returns the latest available calibration results as a list of
		dicts, each with the following keys:
		CALX: Calibration point's horizontal coordinate.
		CALY: Calibration point's vertical coordinate
		LX:   Left eye's recorded horizontal point of gaze.
		LY:   Left eye's recorded vertical point of gaze.
		LV:   Left eye's validity status (1=valid, 0=invalid)
		RX:   Right eye's recorded horizontal point of gaze.
		RY:   Right eye's recorded vertical point of gaze.
		RV:   Right eye's validity status (1=valid, 0=invalid)
		
		Returns None if no calibration results are available.
		"""
		
		# Parameters of the 'CALIB_RESULT' dict.
		params = ['CALX', 'CALY', 'LX', 'LY', 'LV', 'RX', 'RY', 'RV']
		
		# Return the result.
	
		with self._inlock:
			cal_data = self._incoming.get('CAL', {}).get('CALIB_RESULT')
			if not cal_data:
				return None

			cal = copy.deepcopy(cal_data)
			n_points = (len(cal) - 1) // len(params)  # -1 for 'ID'

			results = [
				{
					param: (cal[f'{param}{i}'] == '1' if param in ['LV', 'RV']
							else float(cal[f'{param}{i}']))
					for param in params
				}
				for i in range(1, n_points + 1)
			]

			return results



	def wait_for_calibration_point_start(self, timeout=10.0):
		
		"""Waits for the next calibration point start, which is defined as
		the first unregistered point after the latest calibration start
		message. This function allows for setting a timeout in seconds
		(default = 10.0). Returns the (x,y) coordinate in relative
		coordinates (proportions of the screen width and height) if the
		point started, and None after a timeout. (Also updates the
		internally stored latest registered calibration point number.)
		"""

		# Get the start time of this function.
		start = time.time()
		
		# Get the most recent calibration start time.
		t0 = None
		while t0 is None and (time.time() - start < timeout):
			with self._inlock:
				t0 = self._incoming.get('ACK', {}).get('CALIBRATE_START', {}).get('t')
				if t0 is None:
					time.sleep(0.001)

		# Return None if there was no calibration start.
		if t0 is None:
			return None
		
		# Wait for a new calibration point start, or a timeout.
		pos = None
		pt_nr = None
		started = False
		timed_out = False
		while not started and not timed_out:
			# Get the latest calibration point start.
			t1 = 0
			data = None
			with self._inlock:
				cal_data = self._incoming.get('CAL', {}).get('CALIB_START_PT')
				if cal_data:
					t1 = cal_data.get('t', 0)
					data = cal_data
			# Check if the point is later than the most recent
			# calibration start.
			if t1 >= t0 and data:
				# Check if the current point is already the latest
				# registered point.
				with self._inlock:
					try:
						pt_nr = int(data.get('PT'))
						x = float(data.get('CALX'))
						y = float(data.get('CALY'))
					except (TypeError, ValueError, KeyError):
						pt_nr, x, y = None, None, None
	

				#if pt_nr != self._current_calibration_point:
				if pt_nr is not None and pt_nr != self._current_calibration_point:
					self._current_calibration_point = pt_nr
					pos = (x, y)
					started = True
			# Check if there is a timeout.
			if time.time() - start > timeout:
				timed_out = True
			# Wait for a short bit to avoid wasting too many resources,
			# and to avoid hogging the incoming messages Lock.
			if not timed_out:
				time.sleep(0.001)
		
		return (pt_nr, pos) if started else (None, None)
	
	
	def user_data(self, value):
		
		"""Set the value of the user data field for embedding custom data
		into the data stream. The user data value should be a string.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'USER_DATA', values=[('VALUE', str(value))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	
	def tracker_display(self, state):
		
		"""Shows (state=1) or hides (state=0) the eye tracker display
		window.
		"""
		# Description: Show or hide the eye-tracker display window
        # Parameter: STATE
        # Parameter type: boolean (0 or 1)
        # state (int): 1, aby wyświetlić okno; 0, aby je ukryć.

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'TRACKER_DISPLAY', values=[('STATE', int(state))], wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	

	def get_time_tick_frequency(self):
		
		"""Returns the time-tick frequency to convert the TIME_TICK
		variable to seconds.
		"""
		# Description: Get the time-tick frequency to convert the TIME_TICK variable to seconds
        # Parameter: FREQ
        # Parameter type: longlong 

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('GET', 'TIME_TICK_FREQUENCY', values=None, wait_for_acknowledgement=True)
		
		# Return the result.
		frequency = None
		if acknowledged:
			with self._inlock:
				frequency = copy.copy(self._incoming['ACK']['TIME_TICK_FREQUENCY']['FREQ'])
		
		return frequency
	
	def screen_size(self, x, y, w, h):
		
		"""Set the gaze tracking screen position (x,y) and size (w, h). You
		can use this to work with multi-monitor systems. All values are in
		pixels.
		"""

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'SCREEN_SIZE', 
			values=[('X', x), ('Y', y), ('WIDTH', w), ('HEIGHT', h)], 
			wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	
	

	def get_screen_size(self):
		
		"""Returns the x and y coordinates of the top-left of the screen in
		pixels, as well as the screen width and height in pixels. The
		result is returned as [x, y, w, h].
		"""
		# Description: Get the gaze tracking screen position and size or set the screen on which the gaze
        # tracking is to be performed. Provides the ability to work with multi-monitor systems.

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('GET', 'SCREEN_SIZE', values=None, wait_for_acknowledgement=True)
		
		# Return the result.
		x, y, w, h = None, None, None, None

		if acknowledged:
			with self._inlock:
				x = copy.copy(self._incoming['ACK']['SCREEN_SIZE']['X'])
				y = copy.copy(self._incoming['ACK']['SCREEN_SIZE']['Y'])
				w = copy.copy(self._incoming['ACK']['SCREEN_SIZE']['WIDTH'])
				h = copy.copy(self._incoming['ACK']['SCREEN_SIZE']['HEIGHT'])
		
		return [x, y, w, h]
	

	
	def get_camera_size(self):
		
		"""Returns the size of the camera sensor in pixels, as [w,h].
		"""
		# Description: Get the size of the camera sensor in pixels
        # Parameter: WIDTH (camera width in pixels)
        # Parameter: HEIGHT (camera height in pixels)
        # Parameter type: integer 
        # Returns the size of the camera sensor in pixels, as [w,h]

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('GET', 'CAMERA_SIZE', values=None, wait_for_acknowledgement=True)
		
		# Return the result.
		w = None
		h = None
		if acknowledged:
			with self._inlock:
				w = copy.copy(self._incoming['ACK']['CAMERA_SIZE']['WIDTH'])
				h = copy.copy(self._incoming['ACK']['CAMERA_SIZE']['HEIGHT'])
			
		return [w, h]
	
	def get_product_id(self):
		
		"""Returns the identifier of the connected eye-tracker.
		"""
		# Description: Get the identifier of the current eye-tracker being used
        # Parameter: VALUE (product name)
        # Parameter type: string

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('GET', 'PRODUCT_ID', values=None, wait_for_acknowledgement=True)
		
		# Return the result.
		value = None
		bus = None 
		rate = None
		if acknowledged:
			with self._inlock:
				value = copy.copy(self._incoming['ACK']['PRODUCT_ID']['VALUE'])
				bus = copy.copy(self._incoming['ACK']['PRODUCT_ID']['BUS'])
				rate = copy.copy(self._incoming['ACK']['PRODUCT_ID']['RATE'])
		
		return [value, bus, rate]
	

	
	def get_serial_id(self):
		
		"""Returns the serial number of the connected eye-tracker.
		"""
		# Description: Get the serial number of the eye-tracker
        # Parameter: VALUE (serial number)
        # Parameter type: string

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('GET', 'SERIAL_ID', values=None, wait_for_acknowledgement=True)
		
		# Return the result.
		value = None
		if acknowledged:
			with self._inlock:
				value = copy.copy(self._incoming['ACK']['SERIAL_ID']['VALUE'])
		
		return value
	
	
	def get_company_id(self):
		
		"""Returns the identifier of the manufacturer of the connected
		eye-tracker.
		"""
		# Description: Get the identifier of the eye-tracker manufacturer
        # Parameter: VALUE (company name)
        # Parameter type: string

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('GET', 'COMPANY_ID', values=None, wait_for_acknowledgement=True)
		
		# Return the result.
		value = None
		if acknowledged:
			with self._inlock:
				value = copy.copy(self._incoming['ACK']['COMPANY_ID']['VALUE'])
		
		return value
	
	
	def get_api_id(self):
		
		"""Returns the API version number.
		"""

		# Send the message (returns after the Server acknowledges receipt).
        # Description: Get the API version number  #Pobiera numer wersji API z serwera.
        # Parameter: VALUE (API version)
        # Parameter type: string
		
		acknowledged, timeout = self._send_message('GET', 'API_ID', values=None, wait_for_acknowledgement=True)
		
		# Return the result.
		value = None
		if acknowledged:
			with self._inlock:
				value = copy.copy(self._incoming['ACK']['API_ID']['VALUE'])
        
		return value
	
	def get_tracker_id(self):
		
		"""Get eye-tracker type and id information.
		"""

		# Parameter: ACTIVE_ID (ID number of current active eye-tracker) 
		# Parameter type: integer 
		# Permissions: Read/Write  
		# Parameter: MAX_ID (Maximum number of connected eye-trackers) 
		# Parameter type: integer 
		# Permissions: Read only (ignored when written by a SET command) 
		# Parameter: SEARCH (Automatic eye-tracker search algorithm: NONE=no search,  SIMPLE=move to 
		# next eye-tracker if both eyes lost, CURSOR=move to eye-tracker on screen with current mouse cursor, 
		# GAZE=move to eye-tracker on screen currently targeted by the users gaze). 
		# Parameter type: string 
		# Permissions: Read/Write
		
		acknowledged, timeout = self._send_message('GET', 'TRACKER_ID', values=None, wait_for_acknowledgement=True)

		# Return the result.
		active_id = None
		max_id = None 
		search = None 
		if acknowledged:
			with self._inlock:
				active_id = copy.copy(self._incoming['ACK']['TRACKER_ID']['ACTIVE_ID'])
				max_id = copy.copy(self._incoming['ACK']['TRACKER_ID']['MAX_ID'])
				search = copy.copy(self._incoming['ACK']['TRACKER_ID']['SEARCH'])

		return [active_id, max_id, search]
	
	def marker_pix(self, value, state):
		
		"""Set the size of the marker in millimeters and turn on or off marker tracking. 
		"""

		# Parameter: VALUE (size of marker in millimeters, print one from below and measure one side of the 
		# marker square, larger squares are more easily tracked but are more intrusive).  
		# Parameter type: float 
		# Parameter: STATE (camera height in pixels) 
		# Parameter type: boolean  
		# Permissions: Read/Write

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'MARKER_PIX', 
			values=[('VALUE', value), ('STATE', state)], 
			wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	
	def aac_filter(self, value):
		
		"""Set/Get the AAC moving window average filter length 
		"""

		# Parameter: VALUE (length of moving window average)  
		# Parameter type: integer 
		# Permissions: Read/Write  

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'AAC_FILTER', 
			values=[('VALUE', value)], 
			wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
	
	def ttl_write(self, channel, value):
		
		"""Set/Get the direction (input or output) and write to the state of the TTL ports if output 
		"""

		# Parameter: CHANNEL (channel number of the TTL port 0,1,2,3,4,5,6)  
		# Parameter type: integer 
		# Permissions: Read/Write  
		# Parameter: VALUE (-1 if channel is input, 0 or 1 if channel is output), ignored if a Get command 
		# Parameter type: integer 
		# Permissions: Read/Write

		# Send the message (returns after the Server acknowledges receipt).
		acknowledged, timeout = self._send_message('SET', 'TTL_WRITE', 
			values=[('CHANNEL', channel), ('VALUE', value)], 
			wait_for_acknowledgement=True)
		
		# Return a success Boolean.
		return acknowledged and not timeout
