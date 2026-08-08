package org.example.tridotpnp.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

enum class ProbabilityEditMode {
    Increase,
    Decrease
}

@Composable
fun PermissionRequestContent(onRequestPermission: () -> Unit) {
    val haptic = LocalHapticFeedback.current
    PermissionCard(
        title = "需要相机权限",
        body = "此应用需要访问您的相机来检测亮点。",
        actionLabel = "授予相机权限",
        onAction = {
            haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
            onRequestPermission()
        }
    )
}

@Composable
fun PermissionRationaleContent(onRequestPermission: () -> Unit) {
    val haptic = LocalHapticFeedback.current
    PermissionCard(
        title = "需要相机权限",
        body = "为了实时检测画面中的亮点，我们需要访问您的相机。\n\n没有相机权限，应用将无法正常工作。",
        actionLabel = "授予权限",
        onAction = {
            haptic.performHapticFeedback(HapticFeedbackType.ContextClick)
            onRequestPermission()
        }
    )
}

enum class ThemeMode {
    Light,
    Dark
}

internal data class FlatPalette(
    val appBackground: Color,
    val sidebarBackground: Color,
    val panel: Color,
    val border: Color,
    val accent: Color,
    val textPrimary: Color,
    val textSecondary: Color,
    val textMuted: Color,
    val textOnAccent: Color
)

internal object FlatUiColors {
    private val light = FlatPalette(
        appBackground = Color(0xFFF4F6F8),
        sidebarBackground = Color(0xFFEAEFF3),
        panel = Color(0xFFFFFFFF),
        border = Color(0xFFD6DEE6),
        accent = Color(0xFF147EFB),
        textPrimary = Color(0xFF122033),
        textSecondary = Color(0xFF536276),
        textMuted = Color(0xFF738296),
        textOnAccent = Color(0xFFF8FBFF)
    )

    private val dark = FlatPalette(
        appBackground = Color(0xFF0F1722),
        sidebarBackground = Color(0xFF16202D),
        panel = Color(0xFF1B2635),
        border = Color(0xFF2B3A4E),
        accent = Color(0xFF56A3FF),
        textPrimary = Color(0xFFF3F7FC),
        textSecondary = Color(0xFFB7C4D4),
        textMuted = Color(0xFF8A9AAF),
        textOnAccent = Color(0xFF08111C)
    )

    private var palette by mutableStateOf(light)

    fun update(themeMode: ThemeMode) {
        palette = if (themeMode == ThemeMode.Dark) dark else light
    }

    val AppBackground get() = palette.appBackground
    val SidebarBackground get() = palette.sidebarBackground
    val Panel get() = palette.panel
    val Border get() = palette.border
    val Accent get() = palette.accent
    val TextPrimary get() = palette.textPrimary
    val TextSecondary get() = palette.textSecondary
    val TextMuted get() = palette.textMuted
    val TextOnAccent get() = palette.textOnAccent
}

internal object FlatUiShapes {
    val Panel = RoundedCornerShape(4.dp)
    val Control = RoundedCornerShape(4.dp)
}

@Composable
internal fun TriDotFlatTheme(
    themeMode: ThemeMode,
    content: @Composable () -> Unit
) {
    FlatUiColors.update(themeMode)
    MaterialTheme(
        colorScheme = if (themeMode == ThemeMode.Dark) {
            androidx.compose.material3.darkColorScheme(
                background = FlatUiColors.AppBackground,
                surface = FlatUiColors.Panel,
                primary = FlatUiColors.Accent,
                onPrimary = FlatUiColors.TextOnAccent,
                onSurface = FlatUiColors.TextPrimary
            )
        } else {
            lightColorScheme(
            background = FlatUiColors.AppBackground,
            surface = FlatUiColors.Panel,
            primary = FlatUiColors.Accent,
            onPrimary = FlatUiColors.TextOnAccent,
            onSurface = FlatUiColors.TextPrimary
            )
        },
        content = content
    )
}

@Composable
internal fun FlatPanel(content: @Composable () -> Unit) {
    Surface(
        color = FlatUiColors.Panel,
        shape = FlatUiShapes.Panel,
        border = BorderStroke(1.dp, FlatUiColors.Border),
        shadowElevation = 0.dp
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            content()
        }
    }
}

@Composable
internal fun FlatIconActionButton(
    onClick: () -> Unit,
    icon: @Composable () -> Unit,
    modifier: Modifier = Modifier
) {
    IconButton(
        modifier = modifier,
        colors = IconButtonDefaults.iconButtonColors(
            containerColor = FlatUiColors.Panel,
            contentColor = FlatUiColors.TextPrimary
        ),
        shape = FlatUiShapes.Control,
        onClick = onClick
    ) {
        icon()
    }
}

@Composable
internal fun ThemeModeButton(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Button(
        onClick = onClick,
        modifier = modifier,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (selected) FlatUiColors.Accent else FlatUiColors.Panel,
            contentColor = if (selected) FlatUiColors.TextOnAccent else FlatUiColors.TextPrimary
        ),
        border = if (selected) null else BorderStroke(1.dp, FlatUiColors.Border),
        shape = FlatUiShapes.Control,
        elevation = ButtonDefaults.buttonElevation(defaultElevation = 0.dp)
    ) {
        Text(label)
    }
}

@Composable
internal fun PermissionCard(
    title: String,
    body: String,
    actionLabel: String,
    onAction: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(FlatUiColors.AppBackground)
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Surface(
            color = FlatUiColors.Panel,
            shape = FlatUiShapes.Panel,
            border = BorderStroke(1.dp, FlatUiColors.Border),
            shadowElevation = 0.dp
        ) {
            Column(
                modifier = Modifier
                    .width(320.dp)
                    .padding(24.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = title,
                    style = MaterialTheme.typography.headlineMedium,
                    color = FlatUiColors.TextPrimary,
                    textAlign = TextAlign.Center
                )
                Text(
                    text = body,
                    style = MaterialTheme.typography.bodyLarge,
                    color = FlatUiColors.TextSecondary,
                    textAlign = TextAlign.Center
                )
                Button(
                    onClick = onAction,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = FlatUiColors.Accent,
                        contentColor = FlatUiColors.TextOnAccent
                    ),
                    shape = FlatUiShapes.Control,
                    elevation = ButtonDefaults.buttonElevation(defaultElevation = 0.dp)
                ) {
                    Text(actionLabel)
                }
            }
        }
    }
}







