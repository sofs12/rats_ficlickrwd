/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *  FIClickRwd_EventCodes.h - Header file for this task's Event Codes    *
 *  Created by Sofia Freitas, Oct 2020                                   *
 *  Champalimaud Neuroscience Programme, Paton Lab                       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef FICLickRwd_EventCodes_h
#define FIClickRwd_EventCodes_h

	// States (20's)

	const unsigned int SESSION_PREP = 21;
	const unsigned int BLOCK_PREP = 215;
	const unsigned int TRIAL_PREP = 22;
	const unsigned int WAIT_FI = 23;
	const unsigned int WAIT_FOR_PRESS = 24;
	const unsigned int TIMEOUT_PRESS = 25;
	const unsigned int REWARD_DELIVERY_CUE_SOUND = 26;
	const unsigned int REWARD_DELIVERY_VALVE_OPEN = 27;
	const unsigned int WAIT_ITI = 28;
	const unsigned int WAIT_FOR_PRESS_LEVER_RELEASED = 29; //before the machine goes to the wait for lever press it checks if the lever is not being pressed
	const unsigned int WAIT_FOR_POKE = 245;
	const unsigned int DELTA_CLICK = 235;
	// Timestamp Events
	// (sames as old codes)
	const unsigned int SESSION_START = 70;
	const unsigned int SYNC_PULSE_ON = 61;
	const unsigned int SYNC_PULSE_OFF = 62;
	const unsigned int JITTER_START = 37;
	const unsigned int JITTER_STOP = 38;
	const unsigned int FI_CHANGED = 60;
	const unsigned int RWD_CHANGED = 63;
	const unsigned int BLOCK_CHANGED = 64;
	const unsigned int BLOCK_LEN = 65;

	// Hardware outputs (00's)
	const unsigned int PK_LED_ON = 1;
	const unsigned int PK_LED_OFF = 2;
	const unsigned int VALVE_ON = 3;
	const unsigned int VALVE_OFF = 4;

	const unsigned int PUMP_ON = 5;
	const unsigned int PUMP_OFF = 6;



	// Hardware inputs (10's)
	const unsigned int POKE_IN = 10;
	const unsigned int POKE_OUT = 11;
	const unsigned int LEVER_PRESSED = 12;
	const unsigned int LEVER_RELEASED = 13;
	const unsigned int LICK_IN = 14;
	const unsigned int LICK_OUT = 15;

	// Auditory events (40's)
	const unsigned int ERROR_TIMEOUT_PRESS = 40; // plays when the animal fails to press
	const unsigned int ERROR_TIMEOUT_REWARD = 41; // plys when the animal fails to consume reward
	const unsigned int CLICK_ON = 42; // plays at the end of the fi (when the session has the click condition as true)
	const unsigned int CORRECT_TONE = 43; // plays when a lever press results in reward (secondary reinforcer)


	// Non-timestamp Events

	const unsigned int TRAINING_STAGE = 148;
	const unsigned int RANDOM_SEED = 114; // not in use

	// Session variables (30's)
	const unsigned int FI_SESSION = 30;
	const unsigned int CLICK_SESSION = 31;
	const unsigned int REWARD_RATE_SESSION = 32;
	const unsigned int REWARD_MODIFIER_SESSION = 33;

	// counter for number of protocols given by the syringe pump
	const unsigned int N_PROTOCOL = 34;
	// for how long the valve stays open
	const unsigned int TIME_VALVE = 35;


#endif
