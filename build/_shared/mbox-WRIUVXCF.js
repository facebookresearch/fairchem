import {
  __esm
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/@codemirror/legacy-modes/mode/mbox.js
function styleForHeader(header2) {
  if (header2 === "Subject")
    return "header";
  return "string";
}
function readToken(stream, state) {
  if (stream.sol()) {
    state.inSeparator = false;
    if (state.inHeader && stream.match(whitespace)) {
      return null;
    } else {
      state.inHeader = false;
      state.header = null;
    }
    if (stream.match(separator)) {
      state.inHeaders = true;
      state.inSeparator = true;
      return "atom";
    }
    var match;
    var emailPermitted = false;
    if ((match = stream.match(rfc2822HeaderNoEmail)) || (emailPermitted = true) && (match = stream.match(rfc2822Header))) {
      state.inHeaders = true;
      state.inHeader = true;
      state.emailPermitted = emailPermitted;
      state.header = match[1];
      return "atom";
    }
    if (state.inHeaders && (match = stream.match(header))) {
      state.inHeader = true;
      state.emailPermitted = true;
      state.header = match[1];
      return "atom";
    }
    state.inHeaders = false;
    stream.skipToEnd();
    return null;
  }
  if (state.inSeparator) {
    if (stream.match(email))
      return "link";
    if (stream.match(untilEmail))
      return "atom";
    stream.skipToEnd();
    return "atom";
  }
  if (state.inHeader) {
    var style = styleForHeader(state.header);
    if (state.emailPermitted) {
      if (stream.match(bracketedEmail))
        return style + " link";
      if (stream.match(untilBracketedEmail))
        return style;
    }
    stream.skipToEnd();
    return style;
  }
  stream.skipToEnd();
  return null;
}
var rfc2822, rfc2822NoEmail, whitespace, separator, rfc2822Header, rfc2822HeaderNoEmail, header, email, untilEmail, bracketedEmail, untilBracketedEmail, mbox;
var init_mbox = __esm({
  "../../node_modules/@codemirror/legacy-modes/mode/mbox.js"() {
    rfc2822 = [
      "From",
      "Sender",
      "Reply-To",
      "To",
      "Cc",
      "Bcc",
      "Message-ID",
      "In-Reply-To",
      "References",
      "Resent-From",
      "Resent-Sender",
      "Resent-To",
      "Resent-Cc",
      "Resent-Bcc",
      "Resent-Message-ID",
      "Return-Path",
      "Received"
    ];
    rfc2822NoEmail = [
      "Date",
      "Subject",
      "Comments",
      "Keywords",
      "Resent-Date"
    ];
    whitespace = /^[ \t]/;
    separator = /^From /;
    rfc2822Header = new RegExp("^(" + rfc2822.join("|") + "): ");
    rfc2822HeaderNoEmail = new RegExp("^(" + rfc2822NoEmail.join("|") + "): ");
    header = /^[^:]+:/;
    email = /^[^ ]+@[^ ]+/;
    untilEmail = /^.*?(?=[^ ]+?@[^ ]+)/;
    bracketedEmail = /^<.*?>/;
    untilBracketedEmail = /^.*?(?=<.*>)/;
    mbox = {
      name: "mbox",
      startState: function() {
        return {
          // Is in a mbox separator
          inSeparator: false,
          // Is in a mail header
          inHeader: false,
          // If bracketed email is permitted. Only applicable when inHeader
          emailPermitted: false,
          // Name of current header
          header: null,
          // Is in a region of mail headers
          inHeaders: false
        };
      },
      token: readToken,
      blankLine: function(state) {
        state.inHeaders = state.inSeparator = state.inHeader = false;
      },
      languageData: {
        autocomplete: rfc2822.concat(rfc2822NoEmail)
      }
    };
  }
});
init_mbox();
export {
  mbox
};
//# sourceMappingURL=/build/_shared/mbox-WRIUVXCF.js.map
