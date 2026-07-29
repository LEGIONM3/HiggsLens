import React, { useState, useEffect } from 'react';
import { CernEventObservatory } from './observatory/components/CernEventObservatory';

export const App: React.FC = () => {
  const [selectedEventId, setSelectedEventId] = useState<number>(100001);

  // Check window URL location for permalinks (/display/:eventId or /studio/event/:id or ?eventId=...)
  useEffect(() => {
    const path = window.location.pathname;
    const match = path.match(/\/(display|studio\/event)\/(\d+)/);
    if (match && match[2]) {
      const parsedId = parseInt(match[2], 10);
      if (!isNaN(parsedId)) {
        setSelectedEventId(parsedId);
      }
    } else {
      const searchParams = new URLSearchParams(window.location.search);
      const queryId = searchParams.get('eventId');
      if (queryId) {
        const parsedQueryId = parseInt(queryId, 10);
        if (!isNaN(parsedQueryId)) {
          setSelectedEventId(parsedQueryId);
        }
      }
    }
  }, []);

  const handleEventChange = (eventId: number) => {
    setSelectedEventId(eventId);
    if (typeof window !== 'undefined') {
      window.history.pushState({}, '', `/display/${eventId}`);
    }
  };

  return (
    <CernEventObservatory
      initialEventId={selectedEventId}
      onEventChange={handleEventChange}
    />
  );
};
