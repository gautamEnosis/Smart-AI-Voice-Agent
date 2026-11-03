import { useCallback, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { AlertCircle, ArrowUpRight, Headphones, LockKeyhole, MessageSquareText, PhoneOff, Radio, RefreshCw, ShieldCheck, Sparkles } from 'lucide-react';

import { fetchConsumerConfiguration } from './api';
import { AudioVisualizer } from './components/AudioVisualizer';
import { DualInputBar } from './components/DualInputBar';
import { LiveCaptions } from './components/LiveCaptions';
import { BookingStepper } from './components/generative/BookingStepper';
import { CheckoutModal } from './components/generative/CheckoutModal';
import { RoomCard } from './components/generative/RoomCard';
import { useVoiceSession } from './hooks/useVoiceSession';
import type { BookingStepState, CaptionLine, RoomCardModel, ServerUiEvent } from './types';

const tenantFromUrl = new URLSearchParams(window.location.search).get('tenant') ?? 'grand-meridian';
const embedded = new URLSearchParams(window.location.search).get('embed') === '1';

/** The responsive, WebRTC-synchronized Grand Meridian consumer concierge portal. */
export function App() {
  const captionSequence = useRef(0);
  const [captions, setCaptions] = useState<CaptionLine[]>([]);
  const [rooms, setRooms] = useState<RoomCardModel[]>([]);
  const [selectedRoomId, setSelectedRoomId] = useState<string | null>(null);
  const [booking, setBooking] = useState<BookingStepState | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [checkout, setCheckout] = useState<{ amount?: number; currency?: string; description?: string } | null>(null);
  const configuration = useQuery({
    queryKey: ['consumer-configuration', tenantFromUrl],
    queryFn: () => fetchConsumerConfiguration(tenantFromUrl),
    staleTime: 5 * 60_000,
  });

  const onServerEvent = useCallback((event: ServerUiEvent) => {
    if (event.type === 'caption') {
      captionSequence.current += 1;
      setCaptions((current) => [...current.slice(-23), { id: `caption-${captionSequence.current}`, speaker: event.speaker, text: event.text, isFinal: event.is_final ?? true, createdAt: Date.now() }]);
      return;
    }
    if (event.type === 'room_card') {
      setRooms((current) => [...current.filter((room) => room.id !== event.room.id), event.room].slice(-4));
      return;
    }
    if (event.type === 'booking_stepper') {
      setBooking(event.booking);
      return;
    }
    if (event.type === 'checkout_prompt') {
      setCheckout({ amount: event.amount, currency: event.currency, description: event.description });
      return;
    }
    if (event.type === 'session_ready') {
      setNotice(event.message);
      return;
    }
    if (event.type === 'ptt_acknowledged') {
      setNotice(event.active ? 'Push to Talk is active. Release the button when you finish speaking.' : 'Push to Talk is paused.');
      return;
    }
    setNotice(event.message);
  }, []);

  const voice = useVoiceSession({
    tenantId: configuration.data?.tenant_id ?? tenantFromUrl,
    persona: configuration.data?.persona,
    onServerEvent,
  });
  const connected = voice.status === 'connected';
  const statusCopy = useMemo(() => statusDetails(voice.status, configuration.data?.agent_name ?? 'Elena'), [configuration.data?.agent_name, voice.status]);

  const selectRoom = (room: RoomCardModel) => {
    if (voice.sendEvent({ type: 'client_touch_override', room_id: room.id, label: room.name })) {
      setSelectedRoomId(room.id);
      setNotice(`${room.name} is selected on screen. Elena will confirm it aloud before progressing.`);
    }
  };
  const sendText = (text: string) => {
    if (voice.sendEvent({ type: 'text_message', text })) {
      captionSequence.current += 1;
      setCaptions((current) => [...current.slice(-23), { id: `caption-${captionSequence.current}`, speaker: 'guest', text, isFinal: true, createdAt: Date.now() }]);
    }
  };
  const askForConfirmation = () => {
    voice.sendEvent({ type: 'request_out_loud_confirmation', subject: booking?.room_name ?? 'this reservation' });
  };
  const requestCheckout = () => voice.sendEvent({ type: 'request_secure_checkout' });

  if (configuration.isLoading) return <LoadingPortal embedded={embedded} />;
  if (configuration.isError || !configuration.data) return <ConfigurationError embedded={embedded} onRetry={() => void configuration.refetch()} />;
  const config = configuration.data;

  return (
    <main className={`min-h-screen bg-halo px-4 py-4 text-slate-950 sm:px-6 sm:py-6 ${embedded ? 'min-w-[320px]' : ''}`}>
      <audio autoPlay ref={voice.remoteAudioRef} />
      <div className={`mx-auto ${embedded ? 'max-w-5xl' : 'max-w-7xl'}`}>
        {!embedded && <header className="mb-5 flex flex-col justify-between gap-4 rounded-[1.7rem] border border-white/90 bg-white/75 px-5 py-4 shadow-ambient backdrop-blur sm:flex-row sm:items-center sm:px-6">
          <div className="flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-2xl bg-gradient-to-br from-slate-950 to-violet-700 text-sm font-bold text-white shadow-lg">GM</span>
            <div><p className="text-sm font-semibold text-slate-950">{config.tenant_name}</p><p className="text-xs text-slate-500">Private digital concierge</p></div>
          </div>
          <div className="flex items-center gap-2 text-xs font-medium text-slate-600"><ShieldCheck className="h-4 w-4 text-emerald-600" /> Encrypted browser-to-concierge connection <span className="mx-1 h-1 w-1 rounded-full bg-slate-300" /> <span className="text-slate-400">Powered by Smart-AI-Voice-Agent</span></div>
        </header>}
        <section className={`overflow-hidden rounded-[2rem] border border-white/90 bg-white/70 shadow-ambient backdrop-blur-xl ${embedded ? 'rounded-[1.6rem]' : ''}`}>
          <div className="grid lg:grid-cols-[minmax(0,1.08fr)_minmax(24rem,0.92fr)]">
            <div className="relative border-b border-slate-200/90 p-5 sm:p-7 lg:border-b-0 lg:border-r">
              <div className="pointer-events-none absolute inset-x-0 top-0 h-56 bg-[radial-gradient(ellipse_at_top_left,rgba(196,181,253,0.34),transparent_62%)]" />
              <div className="relative">
                <div className="flex flex-wrap items-start justify-between gap-4">
                  <div>
                    <p className="eyebrow">Always at your service</p>
                    <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 sm:text-3xl">Hello, I’m {config.agent_name}.</h1>
                    <p className="mt-2 max-w-md text-sm leading-6 text-slate-600">Your on-screen concierge for a more confident stay. Talk naturally, hold to speak, or type when it is easier.</p>
                  </div>
                  <ConnectionBadge status={voice.status} />
                </div>
                <div className="mt-6 rounded-[1.65rem] bg-slate-950 p-5 text-white shadow-[0_22px_50px_rgba(30,27,75,0.26)]">
                  <div className="flex items-center justify-between gap-4"><div className="flex items-center gap-3"><span className="relative grid h-11 w-11 place-items-center rounded-2xl bg-white/10"><span className={`absolute inset-0 rounded-2xl ${connected ? 'animate-pulse-ring bg-violet-400' : ''}`} /><Headphones className="relative h-5 w-5" /></span><div><p className="text-sm font-semibold">{statusCopy.title}</p><p className="mt-0.5 text-xs text-slate-300">{statusCopy.detail}</p></div></div>{connected ? <button className="rounded-xl bg-white/10 px-3 py-2 text-xs font-semibold transition hover:bg-white/20" onClick={voice.disconnect} type="button"><PhoneOff className="mr-1 inline h-3.5 w-3.5" />End</button> : <button className="rounded-xl bg-white px-3.5 py-2 text-xs font-bold text-slate-950 transition hover:bg-violet-100 disabled:cursor-wait disabled:opacity-70" disabled={voice.status === 'requesting-permission' || voice.status === 'connecting'} onClick={() => void voice.connect()} type="button"><Radio className="mr-1 inline h-3.5 w-3.5" />Connect</button>}</div>
                  <AudioVisualizer active={connected} stream={voice.microphoneStream} />
                </div>
                {voice.errorMessage && <div className="mt-4 flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 px-3.5 py-3 text-sm leading-5 text-rose-800" role="alert"><AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />{voice.errorMessage}</div>}
                {notice && <div className="mt-4 flex items-start gap-2 rounded-xl border border-violet-100 bg-violet-50/80 px-3.5 py-3 text-sm leading-5 text-violet-900"><Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-violet-600" />{notice}</div>}
                <div className="mt-5"><LiveCaptions captions={captions} /></div>
                <div className="mt-5"><DualInputBar onPushToTalk={voice.setPushToTalk} onSendText={sendText} status={voice.status} /></div>
              </div>
            </div>
            <aside className="bg-slate-50/65 p-5 sm:p-7">
              <div className="flex items-start justify-between gap-4"><div><p className="eyebrow">Live surface</p><h2 className="mt-1 text-xl font-semibold tracking-tight text-slate-950">Your stay, made visible</h2></div><MessageSquareText className="h-5 w-5 text-violet-600" /></div>
              <p className="mt-2 text-sm leading-6 text-slate-600">Recommendations and reservation details appear here as Elena verifies them with you.</p>
              <div className="mt-5 space-y-4">
                {rooms.length === 0 ? <RoomSurfaceEmpty /> : <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">{rooms.map((room) => <RoomCard key={room.id} onSelect={selectRoom} room={room} selected={selectedRoomId === room.id} />)}</div>}
                <BookingStepper booking={booking} onAskForConfirmation={askForConfirmation} />
                <button className="flex w-full items-center justify-between rounded-2xl border border-slate-200 bg-white px-4 py-3.5 text-left text-sm transition hover:border-violet-200 hover:shadow-sm" onClick={() => setCheckout({})} type="button"><span className="flex items-center gap-3"><span className="grid h-9 w-9 place-items-center rounded-xl bg-emerald-50 text-emerald-700"><LockKeyhole className="h-4 w-4" /></span><span><strong className="block text-slate-900">Payment protection</strong><span className="text-xs text-slate-500">A card is never collected in chat or voice.</span></span></span><ArrowUpRight className="h-4 w-4 text-slate-400" /></button>
              </div>
            </aside>
          </div>
        </section>
      </div>
      <CheckoutModal amount={checkout?.amount} currency={checkout?.currency} description={checkout?.description} onClose={() => setCheckout(null)} onRequestSecureCheckout={requestCheckout} open={checkout !== null} paymentCollectionAvailable={config.payment_collection_available} />
    </main>
  );
}

function ConnectionBadge({ status }: { status: ReturnType<typeof useVoiceSession>['status'] }) {
  const connected = status === 'connected';
  const visualUnavailable = status === 'visual-unavailable';
  return <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold ${connected ? 'bg-emerald-50 text-emerald-700' : visualUnavailable ? 'bg-amber-50 text-amber-800' : 'bg-slate-100 text-slate-600'}`}><span className={`h-1.5 w-1.5 rounded-full ${connected ? 'bg-emerald-500' : visualUnavailable ? 'bg-amber-500' : 'bg-slate-400'}`} />{connected ? 'Live voice' : visualUnavailable ? 'Reconnect visual' : 'Ready when you are'}</span>;
}

function RoomSurfaceEmpty() {
  return <section className="overflow-hidden rounded-[1.6rem] border border-dashed border-slate-300 bg-white p-5" aria-label="Room recommendations">
    <div className="flex items-start gap-4"><span className="grid h-11 w-11 shrink-0 place-items-center rounded-2xl bg-gradient-to-br from-violet-100 to-indigo-100 text-violet-700"><Sparkles className="h-5 w-5" /></span><div><h3 className="font-semibold text-slate-900">Personal recommendations will appear here</h3><p className="mt-1 text-sm leading-6 text-slate-600">Once Elena has your dates and preferences, you can compare options visually and tap the one you prefer.</p></div></div>
    <div className="mt-5 grid grid-cols-3 gap-2" aria-hidden="true">{[0, 1, 2].map((index) => <span className="h-1.5 rounded-full bg-slate-100" key={index} />)}</div>
  </section>;
}

function LoadingPortal({ embedded }: { embedded: boolean }) {
  return <main className="grid min-h-screen place-items-center bg-halo p-5"><div className={`w-full animate-pulse rounded-[2rem] border border-white bg-white/80 p-8 shadow-ambient ${embedded ? 'max-w-lg' : 'max-w-xl'}`}><div className="h-3 w-24 rounded bg-slate-200" /><div className="mt-4 h-8 w-3/4 rounded bg-slate-200" /><div className="mt-7 h-36 rounded-3xl bg-slate-200" /><div className="mt-5 h-24 rounded-3xl bg-slate-100" /></div></main>;
}

function ConfigurationError({ embedded, onRetry }: { embedded: boolean; onRetry: () => void }) {
  return <main className="grid min-h-screen place-items-center bg-halo p-5"><section className={`rounded-[2rem] border border-white bg-white p-8 text-center shadow-ambient ${embedded ? 'max-w-lg' : 'max-w-xl'}`}><span className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-rose-50 text-rose-600"><AlertCircle className="h-6 w-6" /></span><h1 className="mt-5 text-xl font-semibold text-slate-950">The concierge is taking a brief pause.</h1><p className="mt-2 text-sm leading-6 text-slate-600">We couldn’t load the secure hotel configuration. Please check the local gateway and try again.</p><button className="mt-6 inline-flex items-center gap-2 rounded-xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white hover:bg-violet-700" onClick={onRetry} type="button"><RefreshCw className="h-4 w-4" /> Try again</button></section></main>;
}

function statusDetails(status: ReturnType<typeof useVoiceSession>['status'], agentName: string) {
  if (status === 'requesting-permission') return { title: 'Waiting for microphone permission', detail: 'Approve access in your browser to begin.' };
  if (status === 'connecting') return { title: 'Creating your private connection', detail: 'Securing a local browser-to-concierge channel…' };
  if (status === 'connected') return { title: 'Your secure voice channel is live', detail: `Hold to speak with ${agentName}; release when you finish.` };
  if (status === 'visual-unavailable') return { title: 'Visual channel paused', detail: 'Voice may continue, but reconnect to restore live cards.' };
  if (status === 'error') return { title: 'Connection needs attention', detail: 'Review the message below and reconnect when ready.' };
  if (status === 'ended') return { title: 'Conversation ended', detail: 'Your microphone is off. Reconnect whenever you need help.' };
  return { title: 'A private concierge is ready', detail: 'Connect when you are ready to speak with Elena.' };
}
